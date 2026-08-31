"""Optional action sections for the weekly email digest (roadmap R12.2).

Pure formatting + best-effort data gathers. Sections omit cleanly when offseason
or when league/player data is unavailable — never fail the whole digest.
"""
from __future__ import annotations

import logging
from html import escape
from typing import Any, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)


def action_section_html(
    title: str,
    body: str,
    *,
    href: str = "",
    cta: str = "Open →",
) -> str:
    """One titled action block for the digest email body."""
    title_s = escape(str(title or "").strip())
    body_s = escape(str(body or "").strip())
    if not title_s or not body_s:
        return ""
    link = ""
    if href:
        link = (
            f'<div style="margin-top:8px;">'
            f'<a href="{escape(href)}" style="font-size:13px;font-weight:700;'
            f'color:#2563eb;text-decoration:none;">{escape(cta)}</a></div>'
        )
    return (
        f'<div style="margin:18px 0 0;padding:12px 14px;border-radius:10px;'
        f'background:#f8fafc;border:1px solid #e2e8f0;">'
        f'<div style="font-size:11px;font-weight:800;letter-spacing:.04em;'
        f'text-transform:uppercase;color:#64748b;">{title_s}</div>'
        f'<div style="margin-top:4px;font-size:14px;color:#0f172a;line-height:1.45;">{body_s}</div>'
        f"{link}</div>"
    )


def player_deep_link(
    base: str,
    platform: str,
    season: int,
    league_id: str,
    pid: str,
    name: str = "",
) -> str:
    """Dashboard URL that opens the player modal via ``?player=``."""
    url = (
        f"{base.rstrip('/')}/{platform}/{int(season)}/{league_id}/dashboard"
        f"?player={quote(str(pid), safe='')}"
    )
    nm = (name or "").strip()
    if nm:
        url += f"&player_name={quote(nm)}"
    return url


def lineup_digest_note(issues: list[dict]) -> Optional[dict[str, str]]:
    """Return ``{title, body}`` for the worst lineup issue, or None."""
    if not issues:
        return None
    try:
        from utils.lineup_issues import summarize_issues
        summary = summarize_issues(issues)
    except Exception:
        summary = ""
    if not summary:
        return None
    kinds = {str(i.get("kind") or "") for i in issues}
    if "empty" in kinds:
        title = "Start/Sit · empty slot"
    elif "injury" in kinds:
        title = "Start/Sit · injured starter"
    elif "bye" in kinds:
        title = "Start/Sit · bye week"
    else:
        title = "Start/Sit"
    return {"title": title, "body": summary}


def top_waiver_from_values(
    model_rows: list[dict],
    owned_ids: set[str],
    *,
    min_value: float = 40.0,
) -> Optional[dict[str, Any]]:
    """Pick the highest-value unowned skill player from a model value table."""
    best = None
    best_val = -1.0
    for row in model_rows or []:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("id") or row.get("player_id") or "").strip()
        if not pid or pid in owned_ids:
            continue
        pos = str(row.get("pos") or row.get("position") or "").upper()
        if pos in ("K", "DEF", "DST"):
            continue
        try:
            val = float(row.get("value") or 0)
        except (TypeError, ValueError):
            val = 0.0
        if val < min_value:
            continue
        if val > best_val:
            best_val = val
            name = (
                row.get("name")
                or row.get("full_name")
                or row.get("player")
                or pid
            )
            best = {
                "player_id": pid,
                "name": str(name),
                "pos": pos,
                "value": val,
            }
    return best


def gather_digest_actions(
    *,
    platform: str,
    season: int,
    league_id: str,
    roster: dict,
    pidx: dict,
    base_url: str,
) -> list[str]:
    """Best-effort HTML action sections for one digest recipient.

    Returns an empty list when offseason / no useful actions. Never raises.
    """
    out: list[str] = []
    try:
        from dashboard_services.api import get_nfl_state
        nfl = get_nfl_state() or {}
    except Exception:
        try:
            from app import get_nfl_state
            nfl = get_nfl_state() or {}
        except Exception:
            nfl = {}

    season_type = str(nfl.get("season_type") or "")
    week = int(nfl.get("week") or 0)
    in_season = season_type in ("reg", "post") and week > 0

    plat = (platform or "sleeper").strip().lower()
    lid = str(league_id or "").strip()
    base = (base_url or "").rstrip("/")
    waivers_url = f"{base}/{plat}/{int(season)}/{lid}/waivers"
    startsit_url = f"{waivers_url}?tab=startsit"

    owned = {str(p) for p in (roster.get("players") or [])}
    starters = [str(p) for p in (roster.get("starters") or [])]

    # ── Start/Sit note (in-season only) ─────────────────────────────────────
    if in_season and starters:
        try:
            from utils.lineup_issues import find_lineup_issues
            from utils.utils import load_week_schedule
            try:
                from dashboard_services.api import get_nfl_players
                nfl_players = get_nfl_players() or {}
            except Exception:
                nfl_players = {}
            teams_playing: set[str] = set()
            try:
                for g in load_week_schedule(int(nfl.get("season") or season), week) or []:
                    for side in ("home", "away"):
                        t = str(g.get(side) or "").upper()
                        if t:
                            teams_playing.add(t)
            except Exception:
                teams_playing = set()
            info = {}
            for pid in starters:
                pl = nfl_players.get(pid) or pidx.get(pid) or {}
                info[pid] = {
                    "name": pl.get("full_name") or pl.get("name") or pl.get("last_name") or "",
                    "team": pl.get("team") or "",
                    "injury_status": pl.get("injury_status") or "",
                }
            issues = find_lineup_issues(starters, info, teams_playing or None)
            note = lineup_digest_note(issues)
            if note:
                html = action_section_html(
                    note["title"], note["body"], href=startsit_url, cta="Fix lineup →",
                )
                if html:
                    out.append(html)
        except Exception:
            logger.debug("[digest-actions] lineup note failed", exc_info=True)

    # ── Top waiver target (value heuristic; omit if nothing material) ───────
    try:
        import time as _time
        from app import DASHBOARD_CACHE, CACHE_TTL, _cache_key
        key = _cache_key(plat, int(season), lid)
        entry = DASHBOARD_CACHE.get(key) or {}
        ctx = {}
        if entry and (_time.time() - float(entry.get("ts") or 0) <= float(CACHE_TTL or 0)):
            ctx = entry.get("ctx") or {}
        rows = (ctx.get("model_value_table") or []) if ctx else []
        # Also treat every rostered player in the league as owned when cached.
        for r in (ctx.get("rosters") or []):
            for p in (r.get("players") or []):
                owned.add(str(p))
        target = top_waiver_from_values(rows, owned) if rows else None
        if target:
            nm = target["name"]
            pos = target.get("pos") or ""
            body = f"Top available by BR value: {nm}"
            if pos:
                body = f"Top available by BR value: {pos} {nm}"
            html = action_section_html(
                "Waiver wire",
                body + f" (≈{int(round(target['value']))} value).",
                href=waivers_url,
                cta="View waivers →",
            )
            if html:
                out.append(html)
    except Exception:
        logger.debug("[digest-actions] waiver target failed", exc_info=True)

    # ── Optional injury approx (only when R07 injury_plan is present) ───────
    if in_season and owned:
        try:
            from utils.injury_plan import injury_plan
            from dashboard_services.injury_return import weeks_out_for_player
            try:
                from dashboard_services.api import get_nfl_players
                nfl_players = get_nfl_players() or {}
            except Exception:
                nfl_players = {}
            for pid in list(owned)[:50]:
                pl = nfl_players.get(pid) or pidx.get(pid) or {}
                st = str(pl.get("injury_status") or "").strip()
                if not st or st.upper() in ("ACTIVE", "ACT", "HEALTHY"):
                    continue
                plan = injury_plan(
                    status=st,
                    espn_weeks=weeks_out_for_player(pid),
                    player_value=None,
                )
                if not plan or plan.get("verdict") not in ("IR", "Drop candidate", "Stash"):
                    continue
                name = pl.get("full_name") or pl.get("name") or "Injured player"
                weeks = plan.get("weeks_label") or "unknown window"
                body = (
                    f"{name}: {plan['verdict']} ({weeks}, approx). "
                    f"{plan.get('reason') or ''}"
                ).strip()
                html = action_section_html(
                    "Injury (approx)",
                    body,
                    href=startsit_url,
                    cta="Review roster →",
                )
                if html:
                    out.append(html)
                break
        except Exception:
            logger.debug("[digest-actions] injury note skipped", exc_info=True)

    return out
