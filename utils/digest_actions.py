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
    title_s = escape(str(title or "").strip(), quote=False)
    body_s = escape(str(body or "").strip(), quote=False)
    if not title_s or not body_s:
        return ""
    link = ""
    if href:
        link = (
            f'<div style="margin-top:8px;">'
            f'<a href="{escape(href, quote=True)}" style="font-size:13px;font-weight:700;'
            f'color:#2563eb;text-decoration:none;">{escape(cta, quote=False)}</a></div>'
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


def value_keys_for_format(fmt: dict) -> tuple[str, str]:
    """Same primary/fallback value columns the in-app waiver surfaces use."""
    is_sf = bool((fmt or {}).get("is_superflex"))
    if (fmt or {}).get("is_redraft") or (fmt or {}).get("is_keeper"):
        return (("redraft_value_sf" if is_sf else "redraft_value_1qb"),
                ("sf_value" if is_sf else "value"))
    return (("sf_value" if is_sf else "value"), "value")


def recommend_waivers(
    model_rows: list[dict],
    owned_ids: set[str],
    *,
    roster_players: Optional[list] = None,
    roster_positions: Optional[list] = None,
    pidx: Optional[dict] = None,
    movers: Optional[dict] = None,
    breakout_by_pid: Optional[dict] = None,
    fmt: Optional[dict] = None,
    limit: int = 3,
    min_score: float = 35.0,
) -> list[dict[str, Any]]:
    """Rank unowned players with ``waiver_pickup_score`` (canonical model).

    Returns up to ``limit`` actionable targets. Empty when nothing clears the
    floor. Does not call the waiver HTTP API.
    """
    fmt = fmt or {}
    pidx = pidx or {}
    owned = {str(p) for p in (owned_ids or set())}
    primary, fallback = value_keys_for_format(fmt)
    need_mults: dict[str, float] = {}
    try:
        from utils.lineup_slots import count_lineup_slots, start_sit_pos
        from utils.waiver_score import need_multiplier, positional_need_scores
        counts: dict[str, int] = {}
        for pid in roster_players or []:
            meta = pidx.get(str(pid)) or {}
            pos = start_sit_pos(meta.get("position") or meta.get("pos") or "")
            if pos:
                counts[pos] = counts.get(pos, 0) + 1
        slots = count_lineup_slots(roster_positions or [])
        starter_reqs = {
            "QB": max(1, int(slots.get("QB") or 0) + int(slots.get("SUPER_FLEX") or 0)),
            "RB": max(1, int(slots.get("RB") or 0) + int(slots.get("FLEX") or 0)),
            "WR": max(1, int(slots.get("WR") or 0) + int(slots.get("FLEX") or 0)),
            "TE": max(1, int(slots.get("TE") or 0)),
        }
        need_scores = positional_need_scores(counts, starter_reqs)
        need_mults = {pos: need_multiplier(pos, need_scores) for pos in starter_reqs}
    except Exception:
        logger.debug("[digest-actions] positional need skipped", exc_info=True)

    delta_by_pid: dict[str, float] = {}
    for bucket in ("risers", "fallers"):
        for m in (movers or {}).get(bucket) or []:
            pid = str(m.get("player_id") or "")
            try:
                if pid:
                    delta_by_pid[pid] = float(m.get("delta") or 0)
            except (TypeError, ValueError):
                continue

    waiver_breakout: dict[str, float] = {}
    for pid, rec in (breakout_by_pid or {}).items():
        try:
            waiver_breakout[str(pid)] = float(
                rec.get("score") if isinstance(rec, dict) else rec or 0
            )
        except (TypeError, ValueError):
            continue

    try:
        from utils.waiver_score import WEIGHTS, waiver_pickup_score, waiver_signal
    except Exception:
        return []

    scored: list[tuple[float, dict]] = []
    for row in model_rows or []:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("id") or row.get("player_id") or "").strip()
        if not pid or pid in owned:
            continue
        pos = str(row.get("pos") or row.get("position") or "").upper()
        if pos in ("K", "DEF", "DST") or not pos:
            continue
        try:
            val = float(row.get(primary) or row.get(fallback) or row.get("value") or 0)
        except (TypeError, ValueError):
            val = 0.0
        if val < float(getattr(WEIGHTS, "min_value", 25.0) or 25.0):
            continue
        age = row.get("age")
        try:
            age_f = float(age) if age is not None else 0
        except (TypeError, ValueError):
            age_f = 0.0
        cand = {
            "player_id": pid,
            "value": val,
            "age": age_f,
            "position": pos,
            "rank_change_7d": delta_by_pid.get(pid, 0.0),
            "need_mult": need_mults.get(pos, 1.0),
        }
        try:
            score = float(waiver_pickup_score(cand, waiver_breakout))
        except Exception:
            continue
        if score < min_score:
            continue
        name = (
            row.get("name") or row.get("full_name") or row.get("player")
            or (pidx.get(pid) or {}).get("full_name")
            or (pidx.get(pid) or {}).get("name")
            or ""
        )
        name = str(name).strip()
        if not name or name == pid:
            continue
        badge = ""
        try:
            _cls, label = waiver_signal(cand, waiver_breakout)
            badge = str(label or "").strip()
        except Exception:
            badge = ""
        reason_bits = []
        if cand["need_mult"] and cand["need_mult"] > 1.05:
            reason_bits.append(f"{pos} need")
        if badge and badge.lower() not in ("target", ""):
            reason_bits.append(badge)
        scored.append((score, {
            "player_id": pid,
            "name": name,
            "pos": pos,
            "value": val,
            "score": score,
            "reason": ", ".join(reason_bits),
        }))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [row for _s, row in scored[: max(0, int(limit or 0))]]


def start_sit_swap_note(
    *,
    starters: list,
    roster: dict,
    pidx: dict,
    nfl_players: dict,
    proj_map: dict,
    roster_positions: list,
    min_gain: float = 2.0,
) -> Optional[dict[str, str]]:
    """Reuse ``projection_upgrades`` — do not invent a second start/sit model."""
    if not starters or not proj_map or not roster_positions:
        return None
    try:
        from utils.lineup_issues import projection_upgrades
    except Exception:
        return None
    reserve = {str(p) for p in (roster.get("reserve") or [])}
    taxi = {str(p) for p in (roster.get("taxi") or [])}
    eligible = [
        str(p) for p in (roster.get("players") or [])
        if str(p) not in reserve and str(p) not in taxi
    ]
    pos_map = {}
    for pid in eligible:
        pl = nfl_players.get(pid) or pidx.get(pid) or {}
        pos_map[pid] = str(pl.get("position") or pl.get("pos") or "")
    try:
        swaps = projection_upgrades(
            [str(p) for p in starters], eligible, proj_map, pos_map,
            list(roster_positions or []), min_gain=min_gain, max_swaps=1,
        )
    except Exception:
        logger.debug("[digest-actions] projection_upgrades failed", exc_info=True)
        return None
    if not swaps:
        return None
    swap = swaps[0]
    pin, pout = str(swap.get("in") or ""), str(swap.get("out") or "")
    name_in = _display_name(pin, nfl_players, pidx)
    name_out = _display_name(pout, nfl_players, pidx)
    if not name_in or not name_out:
        return None
    gain = swap.get("gain")
    try:
        gain_s = f" (+{float(gain):.1f} projected)" if gain is not None else ""
    except (TypeError, ValueError):
        gain_s = ""
    return {
        "title": "Start/Sit",
        "body": f"Consider {name_in} over {name_out}{gain_s}.",
        "in_id": pin,
        "out_id": pout,
    }


def _display_name(pid: str, nfl_players: dict, pidx: dict) -> str:
    pl = (nfl_players or {}).get(pid) or (pidx or {}).get(pid) or {}
    name = str(pl.get("full_name") or pl.get("name") or pl.get("last_name") or "").strip()
    if not name or name == pid or name.lower().startswith("player "):
        return ""
    return name


def gather_digest_actions(
    *,
    platform: str,
    season: int,
    league_id: str,
    roster: dict,
    pidx: dict,
    base_url: str,
    fmt: Optional[dict] = None,
    owned_ids: Optional[set] = None,
    model_rows: Optional[list] = None,
    movers: Optional[dict] = None,
    nfl_state: Optional[dict] = None,
    nfl_players: Optional[dict] = None,
    teams_playing: Optional[set] = None,
    proj_map: Optional[dict] = None,
    roster_positions: Optional[list] = None,
    breakout_by_pid: Optional[dict] = None,
) -> list[str]:
    """Best-effort HTML action sections for one digest recipient.

    Returns an empty list when offseason / no useful actions. Never raises.
    """
    items = gather_digest_action_items(
        platform=platform, season=season, league_id=league_id, roster=roster,
        pidx=pidx, base_url=base_url, fmt=fmt, owned_ids=owned_ids,
        model_rows=model_rows, movers=movers, nfl_state=nfl_state,
        nfl_players=nfl_players, teams_playing=teams_playing, proj_map=proj_map,
        roster_positions=roster_positions, breakout_by_pid=breakout_by_pid,
    )
    out: list[str] = []
    for item in items:
        html = item.get("html") if isinstance(item, dict) else item
        if html:
            out.append(str(html))
    return out


def gather_digest_action_items(
    *,
    platform: str,
    season: int,
    league_id: str,
    roster: dict,
    pidx: dict,
    base_url: str,
    fmt: Optional[dict] = None,
    owned_ids: Optional[set] = None,
    model_rows: Optional[list] = None,
    movers: Optional[dict] = None,
    nfl_state: Optional[dict] = None,
    nfl_players: Optional[dict] = None,
    teams_playing: Optional[set] = None,
    proj_map: Optional[dict] = None,
    roster_positions: Optional[list] = None,
    breakout_by_pid: Optional[dict] = None,
) -> list[dict]:
    """Structured action items (start/sit, waiver, injury) plus pre-rendered HTML."""
    out: list[dict] = []
    nfl = dict(nfl_state or {})
    if not nfl:
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

    owned = {str(p) for p in (owned_ids or roster.get("players") or [])}
    starters = [str(p) for p in (roster.get("starters") or [])]
    players_feed = nfl_players if nfl_players is not None else {}
    if not players_feed:
        try:
            from dashboard_services.api import get_nfl_players
            players_feed = get_nfl_players() or {}
        except Exception:
            players_feed = {}
    playing = set(teams_playing or [])
    if in_season and not playing:
        try:
            from utils.utils import load_week_schedule
            for g in load_week_schedule(int(nfl.get("season") or season), week) or []:
                for side in ("home", "away"):
                    t = str(g.get(side) or "").upper()
                    if t:
                        playing.add(t)
        except Exception:
            playing = set()

    positions = list(roster_positions or [])
    fmt = fmt or {}

    if in_season and starters and not fmt.get("is_best_ball"):
        try:
            from utils.lineup_issues import find_lineup_issues
            info = {}
            for pid in starters:
                pl = players_feed.get(pid) or pidx.get(pid) or {}
                info[pid] = {
                    "name": pl.get("full_name") or pl.get("name") or pl.get("last_name") or "",
                    "team": pl.get("team") or "",
                    "injury_status": pl.get("injury_status") or "",
                }
            issues = find_lineup_issues(starters, info, playing or None)
            note = lineup_digest_note(issues)
            if not note:
                note = start_sit_swap_note(
                    starters=starters, roster=roster, pidx=pidx,
                    nfl_players=players_feed, proj_map=proj_map or {},
                    roster_positions=positions,
                )
            if note:
                html = action_section_html(
                    note["title"], note["body"], href=startsit_url, cta="Fix lineup →",
                )
                if html:
                    out.append({"kind": "lineup", "html": html, **note, "href": startsit_url})
        except Exception:
            logger.debug("[digest-actions] lineup note failed", exc_info=True)

    try:
        rows = list(model_rows) if model_rows is not None else []
        if model_rows is None:
            import time as _time
            from app import DASHBOARD_CACHE, CACHE_TTL, _cache_key
            key = _cache_key(plat, int(season), lid)
            entry = DASHBOARD_CACHE.get(key) or {}
            ctx = {}
            if entry and (_time.time() - float(entry.get("ts") or 0) <= float(CACHE_TTL or 0)):
                ctx = entry.get("ctx") or {}
            rows = list(ctx.get("model_value_table") or [])
            for r in (ctx.get("rosters") or []):
                for p in (r.get("players") or []):
                    owned.add(str(p))
            if not positions:
                positions = list(ctx.get("roster_positions") or [])
        league_owned = set(owned)
        targets = recommend_waivers(
            rows, league_owned,
            roster_players=list(roster.get("players") or []),
            roster_positions=positions,
            pidx=pidx, movers=movers, breakout_by_pid=breakout_by_pid,
            fmt=fmt, limit=3,
        )
        if not targets and rows:
            hit = top_waiver_from_values(rows, league_owned)
            if hit:
                targets = [hit]
        if targets:
            from utils.digest_sections import waiver_html
            html = waiver_html(targets, href=waivers_url)
            if html:
                out.append({
                    "kind": "waiver", "html": html, "targets": targets, "href": waivers_url,
                })
    except Exception:
        logger.debug("[digest-actions] waiver target failed", exc_info=True)

    if in_season and owned:
        try:
            from utils.injury_plan import injury_plan
            from dashboard_services.injury_return import weeks_out_for_player
            for pid in list(owned)[:50]:
                pl = players_feed.get(pid) or pidx.get(pid) or {}
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
                name = pl.get("full_name") or pl.get("name") or ""
                if not name:
                    continue
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
                    out.append({
                        "kind": "injury", "html": html, "body": body, "href": startsit_url,
                    })
                break
        except Exception:
            logger.debug("[digest-actions] injury note skipped", exc_info=True)

    return out
