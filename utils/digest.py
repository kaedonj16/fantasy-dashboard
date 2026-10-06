"""Consolidated utils module: digest.

weekly digest pipeline (actions, context, sections, cross-league, injury plan)

Merged from: utils/digest_actions.py, utils/digest_context.py, utils/digest_sections.py, utils/cross_league_actions.py, utils/injury_plan.py.
Old import paths keep working via compatibility shims.
"""
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations


# ======================================================================
# From utils/digest_actions.py
# ======================================================================

"""Optional action sections for the weekly email digest (roadmap R12.2).

Pure formatting + best-effort data gathers. Sections omit cleanly when offseason
or when league/player data is unavailable — never fail the whole digest.
"""

import logging
import re
from html import escape
from typing import Any, Optional, Sequence
from urllib.parse import quote

logger = logging.getLogger(__name__)


# White cards on the cool-gray email canvas. Accent = left navy-blue bar for actions.
EMAIL_CARD_STYLE = (
    "margin:16px 0 0;padding:14px 16px;border-radius:12px;"
    "background:#ffffff;border:1px solid #e6ebf2;"
)
EMAIL_CARD_ACCENT_STYLE = EMAIL_CARD_STYLE + "border-left:3px solid #2563eb;"
_EMAIL_KICKER = (
    "font-size:11px;font-weight:800;letter-spacing:.06em;"
    "text-transform:uppercase;color:#334155;"
)
_EMAIL_CTA = (
    "display:inline-block;margin-top:2px;font-size:13px;font-weight:700;"
    "color:#2563eb;text-decoration:none;"
)


_EM_DASH = re.compile(r"\s*\u2014\s*")


def _plain_punct(s: str) -> str:
    """Email copy uses ASCII hyphen, not em/en dashes."""
    return _EM_DASH.sub(" - ", s or "").replace("\u2013", "-")


def section_card(
    title: str,
    inner_html: str,
    *,
    href: str = "",
    cta: str = "",
    accent: bool = True,
) -> str:
    """Email-safe titled card. ``inner_html`` is trusted markup (already escaped)."""
    title_s = escape(_plain_punct(str(title or "")).strip(), quote=False)
    inner = _plain_punct(inner_html or "").strip()
    if not title_s or not inner:
        return ""
    link = ""
    if href and cta:
        link = (
            f'<div style="margin-top:10px;">'
            f'<a class="em-cta-a" href="{escape(href, quote=True)}" style="{_EMAIL_CTA}">'
            f"{escape(cta, quote=False)}</a></div>"
        )
    chrome = EMAIL_CARD_ACCENT_STYLE if accent else EMAIL_CARD_STYLE
    return (
        f'<div class="em-sect" style="{chrome}">'
        f'<div class="em-k" style="{_EMAIL_KICKER}">{title_s}</div>'
        f'<div class="em-t" style="margin-top:8px;font-size:14px;color:#0f172a;line-height:1.5;">{inner}</div>'
        f"{link}</div>"
    )


def action_section_html(
    title: str,
    body: str,
    *,
    href: str = "",
    cta: str = "Open →",
) -> str:
    """One titled action block for the digest email body."""
    body_s = escape(_plain_punct(str(body or "")).strip(), quote=False)
    return section_card(title, body_s, href=href, cta=cta, accent=True)


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
        from utils.lineups import summarize_issues
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
    from utils.trade import format_value_keys
    return format_value_keys(
        is_redraft=bool((fmt or {}).get("is_redraft") or (fmt or {}).get("is_keeper")),
        is_sf=bool((fmt or {}).get("is_superflex")),
    )


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
        from utils.lineups import count_lineup_slots, start_sit_pos
        from utils.waivers import need_multiplier, positional_need_scores
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
        from utils.waivers import WEIGHTS, waiver_pickup_score, waiver_signal
    except Exception:
        return []

    is_rd = bool(fmt.get("is_redraft") or fmt.get("is_keeper"))
    is_sf = bool(fmt.get("is_superflex"))
    try:
        n_teams = int(fmt.get("n_teams") or fmt.get("num_teams") or 12)
    except (TypeError, ValueError):
        n_teams = 12
    min_val = float(getattr(WEIGHTS, "min_value", 25.0) or 25.0)

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
        if val < min_val:
            continue
        age = row.get("age")
        try:
            age_f = float(age) if age is not None else 0
        except (TypeError, ValueError):
            age_f = 0.0
        rk = parse_pos_rank(row.get("pos_rank"), str(row.get("pos_rank_label") or ""))
        # extra_depth=1 need_mult > 1 is "thin", not a starter hole. Approximate
        # a hole when the position still wants a meaningful need bump.
        gap = 1.0 if (need_mults.get(pos, 1.0) or 1.0) >= 1.12 else 0.0
        if not waiver_add_clears_quality_bar(
            pos=pos, pos_rank=rk, value=val, is_redraft=is_rd, is_sf=is_sf,
            n_teams=n_teams, age=age_f, starter_gap=gap, min_value=min_val,
        ):
            continue
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
        elif val >= 40:
            reason_bits.append(f"value {int(round(val))}")
        scored.append((score, {
            "player_id": pid,
            "name": name,
            "pos": pos,
            "value": val,
            "score": score,
            "reason": ", ".join(reason_bits),
        }))
    scored.sort(key=lambda t: t[0], reverse=True)
    return unique_waiver_targets([row for _s, row in scored], limit=limit)


def unique_waiver_targets(targets: list, *, limit: int = 3) -> list:
    """Keep one row per position+primary-reason so the email isn't three identical WRs."""
    picked: list = []
    seen: set[tuple] = set()
    cap = max(0, int(limit or 0))
    for row in targets or []:
        if not isinstance(row, dict):
            continue
        key = (str(row.get("pos") or ""), str(row.get("reason") or "").split(",")[0].strip().lower())
        if picked and key in seen:
            continue
        picked.append(row)
        seen.add(key)
        if len(picked) >= cap:
            break
    return picked


def start_sit_swap_note(
    *,
    starters: list,
    roster: dict,
    pidx: dict,
    nfl_players: dict,
    proj_map: dict,
    roster_positions: list,
    min_gain: float = 2.0,
    season: Optional[int] = None,
    week: Optional[int] = None,
) -> Optional[dict[str, str]]:
    """Reuse ``projection_upgrades`` — do not invent a second start/sit model."""
    if not starters or not proj_map or not roster_positions:
        return None
    try:
        from utils.lineups import locked_teams_for_week, projection_upgrades
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
        # Locked players (game already kicked off) cannot be moved, so keep
        # them out of both sides of the suggestion. Fail-open when the
        # schedule is unavailable.
        locked_pids: set = set()
        if season and week:
            try:
                _locked_teams = locked_teams_for_week(int(season), int(week))
            except Exception:
                _locked_teams = set()
            if _locked_teams:
                locked_pids = {
                    pid for pid in eligible
                    if str((nfl_players.get(pid) or {}).get("team") or "").upper() in _locked_teams
                }
        swaps = projection_upgrades(
            [str(p) for p in starters], eligible, proj_map, pos_map,
            list(roster_positions or []), min_gain=min_gain, max_swaps=1,
            injury_status={
                pid: str((nfl_players.get(pid) or {}).get("injury_status") or "")
                for pid in eligible
            },
            locked_pids=locked_pids,
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
            from utils.data_cache import load_week_schedule
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
            from utils.lineups import find_lineup_issues
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
                    roster_positions=positions, season=season, week=week,
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
        ctx = {}
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
        if targets:
            html = waiver_html(targets, href=waivers_url)
            if html:
                out.append({
                    "kind": "waiver", "html": html, "targets": targets, "href": waivers_url,
                })
    except Exception:
        logger.debug("[digest-actions] waiver target failed", exc_info=True)

    if in_season and owned:
        try:
            from dashboard_services.injury_return import weeks_out_for_player
            reserve_set = {str(p) for p in (roster.get("reserve") or []) if p}
            capacity = ir_capacity(positions, roster.get("reserve") or [],
                                   reserve_slots=(ctx.get("league") or {}).get("reserve_slots"))
            for pid in list(owned)[:50]:
                pl = players_feed.get(pid) or pidx.get(pid) or {}
                st = str(pl.get("injury_status") or "").strip()
                if not st or st.upper() in ("ACTIVE", "ACT", "HEALTHY"):
                    continue
                plan = injury_plan(
                    status=st,
                    espn_weeks=weeks_out_for_player(pid),
                    player_value=None,
                    has_open_ir_slot=capacity["has_open_ir_slot"],
                    already_on_ir=pid in reserve_set,
                )
                if not plan or plan.get("verdict") not in ("Move to IR", "Drop candidate"):
                    continue
                # Already on IR: stash/move tips are not actionable (drop still is).
                if pid in reserve_set and plan.get("verdict") == "Move to IR":
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


# ======================================================================
# From utils/digest_context.py
# ======================================================================

"""Shared data for one weekly-digest run.

Loads recipient-independent datasets once and caches per-league payloads so
users in the same league do not refetch. A failure loading one league does not
abort the run.
"""

import logging
from math import erf, sqrt


# Dynasty market-value noise floor for email (absolute BR value delta).
DYNASTY_MOVE_MIN = 40.0
# Skip leaguewide risers weaker than this.
LEAGUEWIDE_MOVE_MIN = 80.0


def uses_long_term_value(fmt: Optional[dict]) -> bool:
    """Dynasty and keeper both keep players; surface market-value movement."""
    fmt = fmt or {}
    return bool(fmt.get("is_dynasty") or fmt.get("is_keeper"))

_FAILED = object()


class DigestRunCache:
    """In-memory cache for a single ``send_weekly_digests`` invocation."""

    def __init__(self) -> None:
        self.pidx: dict = {}
        self.movers_1qb: dict = {}
        self.movers_sf: dict = {}
        self.model_rows: list = []
        self.model_by_id: dict[str, dict] = {}
        self.nfl_state: dict = {}
        self.nfl_players: dict = {}
        self.week_proj: dict[str, float] = {}
        self.teams_playing: set[str] = set()
        self.breakouts: dict[str, dict] = {}
        self._leagues: dict[tuple, Any] = {}
        self._loaded_shared = False

    def load_shared(self) -> None:
        if self._loaded_shared:
            return
        self._loaded_shared = True
        try:
            from dashboard_services.player_value_history import get_top_movers
            from utils.data_cache import load_players_index
            self.pidx = load_players_index() or {}
            self.movers_1qb = get_top_movers(days=7, limit=2000, league_type="1qb") or {}
            self.movers_sf = get_top_movers(days=7, limit=2000, league_type="sf") or {}
        except Exception:
            logger.debug("[digest-cache] movers/index load failed", exc_info=True)
            self.pidx = self.pidx or {}
            self.movers_1qb = self.movers_1qb or {}
            self.movers_sf = self.movers_sf or {}
        try:
            from utils.data_cache import load_model_value_table
            rows = load_model_value_table() or []
            if isinstance(rows, dict):
                rows = list(rows.values()) if rows and isinstance(next(iter(rows.values()), None), dict) else []
            self.model_rows = [r for r in rows if isinstance(r, dict)]
            self.model_by_id = {
                str(r.get("id") or r.get("player_id") or ""): r
                for r in self.model_rows
                if r.get("id") or r.get("player_id")
            }
        except Exception:
            logger.debug("[digest-cache] model values load failed", exc_info=True)
        try:
            from dashboard_services.api import get_nfl_state, get_nfl_players
            self.nfl_state = get_nfl_state() or {}
            self.nfl_players = get_nfl_players() or {}
        except Exception:
            try:
                from app import get_nfl_state
                self.nfl_state = get_nfl_state() or {}
            except Exception:
                self.nfl_state = {}
        self._load_schedule_and_proj()
        self._load_breakouts()

    def _load_schedule_and_proj(self) -> None:
        nfl = self.nfl_state or {}
        try:
            week = int(nfl.get("week") or 0)
            season = int(nfl.get("season") or 0)
        except (TypeError, ValueError):
            week = season = 0
        if nfl.get("season_type") not in ("reg", "post") or week <= 0 or season <= 0:
            return
        try:
            from utils.data_cache import load_week_schedule
            for g in load_week_schedule(season, week) or []:
                for side in ("home", "away"):
                    t = str(g.get(side) or "").upper()
                    if t:
                        self.teams_playing.add(t)
        except Exception:
            logger.debug("[digest-cache] schedule load failed", exc_info=True)
        try:
            from utils.data_cache import load_week_projection
            from utils.projections import weekly_projection_points
            raw = load_week_projection(season, week) or {}
            proj: dict[str, float] = {}
            for pid, entry in (raw.items() if isinstance(raw, dict) else []):
                pts = weekly_projection_points(raw, pid, None, "")
                if pts is None:
                    continue
                try:
                    proj[str(pid)] = float(pts)
                except (TypeError, ValueError):
                    continue
            self.week_proj = proj
        except Exception:
            logger.debug("[digest-cache] weekly projections load failed", exc_info=True)

    def _load_breakouts(self) -> None:
        try:
            from dashboard_services.breakout_api import get_breakout_candidates
            nfl = self.nfl_state or {}
            season = int(nfl.get("season") or 0) or None
            payload = get_breakout_candidates(season=season, min_score=55.0, limit=25) or {}
            if not payload.get("data_available", True):
                return
            out: dict[str, dict] = {}
            for c in payload.get("candidates") or []:
                pid = str(c.get("player_id") or "").strip()
                if not pid:
                    continue
                score = c.get("breakout_opportunity_score") or c.get("breakout_score")
                hit = c.get("hit_probability")
                try:
                    score_f = float(score) if score is not None else None
                except (TypeError, ValueError):
                    score_f = None
                if score_f is None or score_f < 55:
                    continue
                try:
                    hit_f = float(hit) if hit is not None else None
                except (TypeError, ValueError):
                    hit_f = None
                if hit_f is not None and hit_f < 0.15 and score_f < 70:
                    continue
                name = str(c.get("player_name") or c.get("name") or "").strip()
                out[pid] = {
                    "player_id": pid,
                    "name": name,
                    "score": score_f,
                    "hit_probability": hit_f,
                }
            self.breakouts = out
        except Exception:
            logger.debug("[digest-cache] breakout load failed", exc_info=True)

    def movers_for(self, *, is_superflex: bool) -> dict:
        return self.movers_sf if is_superflex else self.movers_1qb

    def league_bundle(self, platform: str, season: int, league_id: str) -> Optional[dict]:
        plat = (platform or "sleeper").strip().lower()
        lid = str(league_id or "").strip()
        try:
            season_i = int(season)
        except (TypeError, ValueError):
            return None
        if not plat or not lid:
            return None
        key = (plat, season_i, lid)
        if key in self._leagues:
            val = self._leagues[key]
            return None if val is _FAILED else val
        try:
            bundle = _load_league_bundle(plat, season_i, lid, self)
            self._leagues[key] = bundle
            return bundle
        except Exception:
            logger.warning(
                "[digest-cache] league load failed platform=%s season=%s",
                plat, season_i, exc_info=True,
            )
            self._leagues[key] = _FAILED
            return None


def _load_league_bundle(platform: str, season: int, league_id: str, cache: DigestRunCache) -> dict:
    from dashboard_services.platform_api import get_league, get_rosters, get_users
    from utils.league import classify_league_roster_format

    league = get_league(platform, league_id, season) or {}
    rosters = get_rosters(platform, league_id, season) or []
    users = get_users(platform, league_id, season) or []
    fmt = classify_league_roster_format(league=league, platform=platform)
    owned: set[str] = set()
    by_rid: dict[str, dict] = {}
    for r in rosters:
        rid = str(r.get("roster_id") or "")
        by_rid[rid] = r
        for p in r.get("players") or []:
            owned.add(str(p))
    uid_name = {
        str(u.get("user_id")): (
            ((u.get("metadata") or {}).get("team_name") if isinstance(u.get("metadata"), dict) else None)
            or u.get("display_name") or u.get("username") or "Team"
        )
        for u in users
    }
    matchups: list = []
    nfl = cache.nfl_state or {}
    try:
        week = int(nfl.get("week") or 0)
    except (TypeError, ValueError):
        week = 0
    if nfl.get("season_type") in ("reg", "post") and week > 0:
        try:
            from dashboard_services.platform_api import get_matchups
            matchups = get_matchups(platform, league_id, week, season) or []
        except Exception:
            logger.debug("[digest-cache] matchups failed", exc_info=True)
            matchups = []
    return {
        "platform": platform,
        "season": season,
        "league_id": league_id,
        "league": league,
        "rosters": rosters,
        "users": users,
        "format": fmt,
        "owned_ids": owned,
        "roster_by_id": by_rid,
        "uid_name": uid_name,
        "matchups": matchups,
        "week": week,
    }


def in_season(cache: DigestRunCache) -> bool:
    nfl = cache.nfl_state or {}
    try:
        week = int(nfl.get("week") or 0)
    except (TypeError, ValueError):
        week = 0
    return nfl.get("season_type") in ("reg", "post") and week > 0


def team_display_name(roster: dict, uid_name: dict) -> str:
    meta = roster.get("metadata") if isinstance(roster.get("metadata"), dict) else {}
    name = str((meta or {}).get("team_name") or "").strip()
    if name:
        return name
    owner = str(roster.get("owner_id") or "")
    return str(uid_name.get(owner) or "").strip()


def value_column(fmt: dict) -> tuple[str, str]:
    """(primary, fallback) model-value keys — same axes as the waiver surfaces."""
    is_sf = bool(fmt.get("is_superflex"))
    if fmt.get("is_redraft") or fmt.get("is_keeper"):
        return (("redraft_value_sf" if is_sf else "redraft_value_1qb"),
                ("sf_value" if is_sf else "value"))
    return (("sf_value" if is_sf else "value"), "value")


def player_value(row: dict, fmt: dict) -> float:
    primary, fallback = value_column(fmt)
    for key in (primary, fallback, "value"):
        try:
            if row.get(key) is not None:
                return float(row.get(key) or 0)
        except (TypeError, ValueError):
            continue
    return 0.0


def filter_movers(
    items: list,
    *,
    want_positive: bool,
    mine: Optional[set[str]] = None,
    min_abs: float = DYNASTY_MOVE_MIN,
    limit: int = 3,
) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for m in items or []:
        pid = str(m.get("player_id") or "")
        d = m.get("delta")
        if not pid or d is None:
            continue
        if mine is not None and pid not in mine:
            continue
        try:
            delta = float(d)
        except (TypeError, ValueError):
            continue
        if abs(delta) < min_abs:
            continue
        if want_positive and delta <= 0:
            continue
        if not want_positive and delta >= 0:
            continue
        out.append((pid, delta))
        if len(out) >= limit:
            break
    return out


def mover_notes(
    pairs: list[tuple[str, float]],
    *,
    my_pids: set[str],
    model_by_id: dict,
    fmt: dict,
    pidx: dict,
) -> dict[str, str]:
    """Explain movement from roster rank when values exist. Never invent a cause."""
    notes: dict[str, str] = {}
    ranked = []
    for pid in my_pids:
        row = model_by_id.get(pid) or {}
        val = player_value(row, fmt) if row else 0.0
        if val <= 0:
            continue
        pos = str(row.get("pos") or row.get("position") or (pidx.get(pid) or {}).get("position") or "").upper()
        ranked.append((pid, pos, val))
    by_pos: dict[str, list] = {}
    for pid, pos, val in ranked:
        by_pos.setdefault(pos or "?", []).append((pid, val))
    for pos, rows in by_pos.items():
        rows.sort(key=lambda t: t[1], reverse=True)
    for pid, delta in pairs:
        row = model_by_id.get(pid) or {}
        pos = str(row.get("pos") or row.get("position") or (pidx.get(pid) or {}).get("position") or "").upper()
        order = by_pos.get(pos) or []
        idx = next((i for i, t in enumerate(order) if t[0] == pid), None)
        if idx is None:
            continue
        rank_n = idx + 1
        label = f"{pos}{rank_n}" if pos else f"#{rank_n}"
        ahead = None
        if idx + 1 < len(order):
            ahead = order[idx + 1][0] if delta < 0 else None
        behind = order[idx - 1][0] if idx > 0 and delta > 0 else None
        neighbor = behind or ahead
        neighbor_name = ""
        if neighbor:
            meta = pidx.get(neighbor) or {}
            neighbor_name = str(meta.get("full_name") or meta.get("name") or "").strip()
        sign = "+" if delta >= 0 else ""
        if neighbor_name:
            verb = "moved ahead of" if delta > 0 else "fell behind"
            notes[pid] = f"{sign}{delta:.0f} value · now your {label} · {verb} {neighbor_name}"
        else:
            notes[pid] = f"{sign}{delta:.0f} value this week · now your {label} by market value"
    return notes


def matchup_for_roster(bundle: dict, roster_id: str, cache: DigestRunCache) -> Optional[dict]:
    rid = str(roster_id or "")
    rows = [m for m in (bundle.get("matchups") or []) if isinstance(m, dict)]
    mine = next((m for m in rows if str(m.get("roster_id")) == rid), None)
    if mine is None:
        return None
    mid = mine.get("matchup_id")
    opp = next(
        (m for m in rows if str(m.get("roster_id")) != rid and m.get("matchup_id") == mid),
        None,
    )
    if opp is None:
        return None
    opp_roster = (bundle.get("roster_by_id") or {}).get(str(opp.get("roster_id")) or "") or {}
    opp_name = team_display_name(opp_roster, bundle.get("uid_name") or {}) or "Opponent"
    if opp_name.lower().startswith("roster ") or str(opp.get("roster_id") or "") == opp_name:
        opp_name = "Opponent"
    user_starters = [str(p) for p in (mine.get("starters") or []) if p and str(p) not in ("0", "None")]
    opp_starters = [str(p) for p in (opp.get("starters") or []) if p and str(p) not in ("0", "None")]
    if not user_starters:
        roster = (bundle.get("roster_by_id") or {}).get(rid) or {}
        user_starters = [str(p) for p in (roster.get("starters") or []) if p and str(p) not in ("0", "None")]
    if not opp_starters:
        opp_starters = [str(p) for p in (opp_roster.get("starters") or []) if p and str(p) not in ("0", "None")]
    user_proj = _sum_proj(user_starters, cache.week_proj)
    opp_proj = _sum_proj(opp_starters, cache.week_proj)
    out: dict[str, Any] = {
        "opponent_name": opp_name,
        "opponent_roster_id": str(opp.get("roster_id") or ""),
    }
    if user_proj is not None and opp_proj is not None and (user_proj > 0 or opp_proj > 0):
        out["user_proj"] = round(user_proj, 1)
        out["opp_proj"] = round(opp_proj, 1)
        out["margin"] = round(user_proj - opp_proj, 1)
        wp = _win_prob_from_starters(user_starters, opp_starters, cache.week_proj)
        if wp is not None:
            out["win_prob"] = wp
    return out


def _sum_proj(pids: list[str], proj: dict[str, float]) -> Optional[float]:
    if not proj:
        return None
    total = 0.0
    any_hit = False
    for pid in pids:
        if pid in proj:
            any_hit = True
            total += float(proj.get(pid) or 0)
    return total if any_hit else None


def _win_prob_from_starters(starters_a: list[str], starters_b: list[str], proj: dict[str, float]) -> Optional[float]:
    """Same projection-normal model as the in-app weekly recap. None without projs."""
    if not proj:
        return None

    def _stats(pids):
        total = var = 0.0
        hits = 0
        for pid in pids or []:
            if pid not in proj:
                continue
            p = float(proj.get(pid) or 0.0)
            total += p
            sigma = max(0.4 * p, 4.0)
            var += sigma * sigma
            hits += 1
        return total, var, hits

    ta, va, ha = _stats(starters_a)
    tb, vb, hb = _stats(starters_b)
    if ha < 3 or hb < 3:
        return None
    cv = va + vb
    if cv < 1e-6:
        return 0.5 if abs(ta - tb) < 1e-9 else (1.0 if ta > tb else 0.0)
    z = (ta - tb) / (sqrt(cv) * sqrt(2.0))
    return max(0.01, min(0.99, 0.5 * (1.0 + erf(z))))


def trade_insight_for_roster(
    *,
    my_pids: set[str],
    model_by_id: dict,
    fmt: dict,
    roster_positions: list,
    pidx: dict,
) -> Optional[dict]:
    """Compact roster-construction note from positional strength. No fake offers."""
    if not uses_long_term_value(fmt) or not my_pids or not model_by_id:
        return None
    try:
        from utils.lineups import count_lineup_slots
        from utils.trade import weighted_pos_strength
    except Exception:
        return None
    slot_counts = count_lineup_slots(roster_positions or [])
    by_pos: dict[str, list[float]] = {"QB": [], "RB": [], "WR": [], "TE": []}
    for pid in my_pids:
        row = model_by_id.get(pid) or {}
        pos = str(row.get("pos") or row.get("position") or (pidx.get(pid) or {}).get("position") or "").upper()
        if pos not in by_pos:
            continue
        val = player_value(row, fmt)
        if val > 0:
            by_pos[pos].append(val)
    strengths = {}
    for pos, vals in by_pos.items():
        if not vals:
            continue
        strengths[pos] = weighted_pos_strength(vals, pos, slot_counts)
    if len(strengths) < 2:
        return None
    strong_pos = max(strengths, key=strengths.get)
    weak_pos = min(strengths, key=strengths.get)
    if strengths[strong_pos] <= 0 or strong_pos == weak_pos:
        return None
    ratio = strengths[strong_pos] / max(strengths[weak_pos], 1.0)
    if ratio < 1.8:
        return None
    body = (
        f"Your {strong_pos} room is your strongest group by market value; "
        f"{weak_pos} is comparatively thin. Worth a look if you want to rebalance."
    )
    return {"title": "Roster construction", "body": body}


def breakout_for_roster(my_pids: set[str], cache: DigestRunCache, pidx: dict) -> Optional[dict]:
    best = None
    best_score = -1.0
    for pid in my_pids:
        hit = (cache.breakouts or {}).get(pid)
        if not hit:
            continue
        score = float(hit.get("score") or 0)
        if score > best_score:
            name = hit.get("name") or _name(pid, pidx)
            if not name:
                continue
            best_score = score
            best = {**hit, "name": name, "player_id": pid}
    return best


def roster_core(
    my_pids: set[str],
    *,
    model_by_id: dict,
    fmt: dict,
    pidx: dict,
    limit: int = 3,
) -> list[dict]:
    """Top roster players by the same value axis the rest of the site uses."""
    rows: list[tuple[float, dict]] = []
    for pid in my_pids or []:
        meta = (pidx or {}).get(pid) or {}
        name = str(
            meta.get("full_name") or meta.get("name")
            or ((meta.get("first_name") or "") + " " + (meta.get("last_name") or "")).strip()
        ).strip()
        if not name or name == pid or name.lower().startswith("player "):
            continue
        row = (model_by_id or {}).get(pid) or {}
        val = player_value(row, fmt) if row else 0.0
        if val < 40:
            continue
        pos = str(row.get("pos") or row.get("position") or meta.get("position") or "").upper()
        rows.append((val, {"player_id": pid, "name": name, "pos": pos, "value": val}))
    rows.sort(key=lambda t: t[0], reverse=True)
    return [item for _v, item in rows[: max(0, int(limit or 0))]]


def _name(pid: str, pidx: dict) -> str:
    meta = (pidx or {}).get(str(pid)) or {}
    return str(
        meta.get("full_name") or meta.get("name")
        or ((meta.get("first_name") or "") + " " + (meta.get("last_name") or "")).strip()
    ).strip()


# ======================================================================
# From utils/digest_sections.py
# ======================================================================

"""Reusable email-safe HTML sections for the weekly digest.

Keep markup table-light, inline-styled, and ~600px wide so it survives mobile
clients. Sections return "" when they have nothing useful to show.
"""


MAX_WIDTH_PX = 600


# Dark-mode theme for email clients. Light-mode colors stay inline (the
# default); these rules override them when the client reports dark mode.
# Apple Mail / iOS Mail honor prefers-color-scheme; the Outlook apps stamp
# [data-ogsc] on the markup instead, so the same rules ship under both.
_DARK_MODE_RULES: tuple = (
    (".em-wrap", (("background", "#0b1220"),)),
    (".em-card", (("background", "#1a2438"), ("border-color", "#2f3f5c"))),
    (".em-head", (("background", "#0b1220"),)),
    (".em-sub", (("color", "#ffffff"),)),
    (".em-kicker", (("color", "#93c5fd"),)),
    (".em-cbody", (("background", "#1a2438"),)),
    (".em-foot", (("background", "#1a2438"), ("border-color", "#2f3f5c"))),
    (".em-foot-t", (("color", "#94a3b8"),)),
    (".em-cta-btn", (("background", "#3b82f6"),)),
    (".em-h", (("color", "#cbd5e1"),)),
    (".em-greet", (("color", "#f1f5f9"),)),
    (".em-t", (("color", "#f1f5f9"),)),
    (".em-t2", (("color", "#cbd5e1"),)),
    (".em-t3", (("color", "#94a3b8"),)),
    (".em-k", (("color", "#cbd5e1"),)),
    (".em-link", (("color", "#60a5fa"),)),
    (".em-cta-a", (("color", "#60a5fa"),)),
    (".em-sect", (("background", "#222f47"), ("border-color", "#2f3f5c"))),
    (".em-chip", (("background", "#1e3a8a"), ("color", "#bfdbfe"))),
    (".em-alert", (("background", "#451a03"), ("border-color", "#92400e"))),
    (".em-alert-k", (("color", "#fbbf24"),)),
    (".em-alert-t", (("color", "#fde68a"),)),
    (".em-up", (("color", "#4ade80"),)),
    (".em-dn", (("color", "#f87171"),)),
    (".em-urgent", (("color", "#7fb0ff"),)),
    (".em-rowb", (("border-top-color", "#2f3f5c"), ("border-bottom-color", "#2f3f5c"))),
)


def _dark_mode_css() -> str:
    """Style block that themes the digest for dark-mode email clients."""
    def block(prefix: str) -> str:
        parts = []
        for sel, decls in _DARK_MODE_RULES:
            body = ";".join(f"{prop}:{val} !important" for prop, val in decls)
            parts.append(f"{prefix}{sel}{{{body};}}")
        return "".join(parts)
    return (
        "<style>"
        "body{margin:0 !important;padding:0 !important;}"
        f"@media (prefers-color-scheme:dark){{{block('')}}}"
        f"{block('[data-ogsc] ')}"
        "</style>"
    )


def email_shell(
    inner_html: str,
    *,
    subtitle: str,
    dash_url: str = "",
    cta_label: str = "Open your dashboard →",
    unsub_href: str = "{UNSUB}",
    logo_url: str = "",
    brand_mark_url: str = "",
    footer_kind: str = "weekly_digest",
    header_theme: str = "dark",
    preheader: str = "",
) -> str:
    """Wrap email body in the BR Fantasy chrome.

    ``footer_kind`` controls unsubscribe copy:
      - ``weekly_digest`` (default)
      - ``onboarding``: signup / PRO welcome emails
      - ``billing``: transactional account notices (dunning, trial expiry,
        win-back). No unsubscribe link: these are required billing notices,
        not marketing.

    ``header_theme`` controls the masthead:
      - ``dark`` (default): navy header with the light distressed wordmark.
      - ``light``: white header with the full-color navy wordmark and no
        redundant kicker line (pair with light-mode logo assets).

    ``preheader`` is hidden inbox preview text. It must be the first text in
    the HTML body; zero-width padding keeps clients from pulling body copy
    into the snippet.

    The shell returns a full HTML document with ``color-scheme`` meta tags
    and a dark-mode stylesheet, so clients that support it (Apple Mail,
    Outlook apps) render an intentional dark theme instead of smart-inverting
    the light one.
    """
    pre = (preheader or "").strip()
    pre_html = ""
    if pre:
        pad = "&#8203;&nbsp;" * 40
        pre_html = (
            '<div style="display:none;max-height:0;overflow:hidden;opacity:0;'
            'color:transparent;visibility:hidden;mso-hide:all;">'
            f"{escape(pre, quote=False)}{pad}</div>"
        )
    sub = escape(subtitle or "Your weekly fantasy digest", quote=False)
    base_logo = (logo_url or "").strip()
    mark = (brand_mark_url or "").strip()
    logo_block = ""
    if base_logo or mark:
        imgs = []
        if mark:
            imgs.append(
                f'<img src="{escape(mark, quote=True)}" alt="" width="36" height="36" '
                f'style="display:block;border:0;outline:none;width:36px;height:36px;'
                f'border-radius:8px;" />'
            )
        if base_logo:
            imgs.append(
                f'<img src="{escape(base_logo, quote=True)}" alt="BR Fantasy" width="160" '
                f'style="display:block;border:0;outline:none;width:160px;height:auto;'
                f'max-width:70%;" />'
            )
        logo_block = (
            '<div style="margin:0 0 14px;">'
            + (
                f'<table role="presentation" cellpadding="0" cellspacing="0"><tr>'
                f'<td style="vertical-align:middle;padding-right:12px;">{imgs[0]}</td>'
                f'<td style="vertical-align:middle;">{imgs[1] if len(imgs) > 1 else ""}</td>'
                f"</tr></table>"
                if len(imgs) == 2
                else imgs[0]
            )
            + "</div>"
        )
    # Palette. "light" matches the site theme (navy --accent #122d4b on a
    # white masthead); "dark" keeps the original weekly-digest chrome so that
    # email is unchanged.
    if header_theme == "light":
        wrapper_bg = "#f4f6f8"
        card_border = "#e8eef4"
        header_bg = "#ffffff"
        subtitle_color = "#122d4b"
        divider = "#122d4b"
        body_bg = "#f8fafc"
        cta_bg = "#122d4b"
        footer_border = "#e8eef4"
        footer_color = "#64748b"
        kicker_html = ""
    else:
        wrapper_bg = "#e8eef5"
        card_border = "#dbe3ee"
        header_bg = "#0b1220"
        subtitle_color = "#ffffff"
        divider = "#2563eb"
        body_bg = "#f7f9fc"
        cta_bg = "#2563eb"
        footer_border = "#e6ebf2"
        footer_color = "#94a3b8"
        kicker_html = (
            '<div class="em-kicker" style="color:#93c5fd;font-size:11px;font-weight:800;'
            'letter-spacing:.14em;text-transform:uppercase;">BR Fantasy</div>'
        )
    cta = ""
    if dash_url:
        cta = f"""\
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin-top:24px;">
  <tr>
    <td>
      <a class="em-cta-btn" href="{escape(dash_url, quote=True)}" style="display:block;background:{cta_bg};color:#ffffff;text-decoration:none;font-weight:700;font-size:15px;padding:14px 20px;border-radius:10px;text-align:center;">{escape(cta_label, quote=False)}</a>
    </td>
  </tr>
</table>"""
    if footer_kind == "billing":
        footer = (
            "You're getting this because you have (or had) BR Fantasy PRO. "
            "This is a billing notice, not marketing email."
        )
    elif footer_kind == "onboarding":
        footer = (
            "You're getting this because you created a BR Fantasy account "
            "(or upgraded to PRO). "
            f'<a href="{escape(unsub_href, quote=True)}" style="color:{footer_color};">Unsubscribe</a> '
            "from welcome and onboarding emails. Weekly digests are separate."
        )
    else:
        footer = (
            "You're getting this because you signed in to BR Fantasy. "
            f'<a href="{escape(unsub_href, quote=True)}" style="color:{footer_color};">Unsubscribe</a> '
            "from weekly digest emails."
        )
    return f"""\
<!DOCTYPE html>
<html lang="en" xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<meta name="supported-color-schemes" content="light dark">
<title>{sub}</title>
{_dark_mode_css()}
</head>
<body style="margin:0;padding:0;word-spacing:normal;">
<div class="em-wrap" style="background:{wrapper_bg};padding:28px 12px;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
  {pre_html}
  <div class="em-card" style="max-width:{MAX_WIDTH_PX}px;margin:0 auto;background:#ffffff;border-radius:16px;overflow:hidden;border:1px solid {card_border};">
    <div class="em-head" style="background:{header_bg};padding:22px 24px 18px;">
      {logo_block}
      {kicker_html}
      <div class="em-sub" style="color:{subtitle_color};font-size:20px;font-weight:800;margin-top:6px;line-height:1.25;">{sub}</div>
    </div>
    <div style="height:4px;background:{divider};line-height:4px;font-size:0;">&nbsp;</div>
    <div class="em-cbody" style="padding:22px 20px 24px;background:{body_bg};">
      {inner_html}
      {cta}
    </div>
    <div class="em-foot" style="padding:16px 22px;background:#ffffff;border-top:1px solid {footer_border};">
      <p class="em-foot-t" style="margin:0;font-size:11px;color:{footer_color};line-height:1.6;">
        {footer}
      </p>
    </div>
  </div>
</div>
</body>
</html>"""


def greeting_html(first_name: Optional[str]) -> str:
    hi = escape(first_name.strip(), quote=False) if first_name and first_name.strip() else "there"
    return (
        f'<p class="em-greet" style="margin:0 0 12px;font-size:16px;color:#0f172a;font-weight:600;">'
        f"Hey {hi},</p>"
    )


def heading(title: str) -> str:
    t = escape(str(title or "").strip(), quote=False)
    if not t:
        return ""
    return (
        f'<h3 class="em-h" style="margin:20px 0 0;font-size:11px;font-weight:800;text-transform:uppercase;'
        f'letter-spacing:.06em;color:#334155;">{t}</h3>'
    )


def format_chip_html(label: str) -> str:
    text = str(label or "").strip()
    if not text:
        return ""
    return (
        f'<span class="em-chip" style="display:inline-block;margin-top:8px;padding:5px 12px;border-radius:999px;'
        f'background:#dbeafe;color:#1d4ed8;font-size:11px;font-weight:700;letter-spacing:.02em;">'
        f"{escape(text, quote=False)}</span>"
    )


def matchup_one_liner(matchup: Optional[dict]) -> str:
    """Compact matchup line for a multi-league overview."""
    if not matchup:
        return ""
    opp = str(matchup.get("opponent_name") or "").strip()
    if not opp:
        return ""
    bits = [f"vs {opp}"]
    wp = matchup.get("win_prob")
    try:
        if wp is not None:
            pct = max(1, min(99, int(round(float(wp) * 100))))
            wpf = float(wp)
            if wpf >= 0.55:
                bits.append(f"Favored {pct}%")
            elif wpf <= 0.45:
                bits.append(f"Underdog {pct}%")
            else:
                bits.append(f"{pct}%")
            return " · ".join(bits)
    except (TypeError, ValueError):
        pass
    you, them = matchup.get("user_proj"), matchup.get("opp_proj")
    try:
        if you is not None and them is not None:
            bits.append(f"{float(you):.0f} to {float(them):.0f}")
    except (TypeError, ValueError):
        pass
    return " · ".join(bits)


def league_focus_line(
    *,
    is_dynasty: bool = False,
    matchup: Optional[dict] = None,
    lineup_note: Optional[dict] = None,
    waiver: Optional[dict] = None,
    injury_body: str = "",
    top_asset: Optional[dict] = None,
    riser_name: str = "",
    riser_delta: Optional[float] = None,
    breakout_name: str = "",
) -> str:
    """One scan-line per league. Actionable items beat flavor."""
    note = lineup_note or {}
    lineup_body = str(note.get("body") or note.get("title") or "").strip()
    if lineup_body:
        return lineup_body
    inj = str(injury_body or "").strip()
    if inj:
        return inj.split(".")[0].strip()[:90]
    if not is_dynasty:
        mu = matchup_one_liner(matchup)
        if mu:
            return mu
    rname = str(riser_name or "").strip()
    if rname:
        try:
            delta = float(riser_delta) if riser_delta is not None else None
        except (TypeError, ValueError):
            delta = None
        extra = f" ▲{abs(delta):.0f}" if delta is not None else ""
        return f"{rname}{extra}"
    asset = top_asset or {}
    aname = str(asset.get("name") or "").strip()
    if aname:
        pos = str(asset.get("pos") or "").upper()
        try:
            val = float(asset.get("value") or 0)
        except (TypeError, ValueError):
            val = 0.0
        bits = [aname]
        if pos:
            bits.append(pos)
        if val >= 40:
            bits.append(f"{val:.0f}")
        return " · ".join(bits)
    wv = waiver or {}
    wname = str(wv.get("name") or "").strip()
    if wname:
        wpos = str(wv.get("pos") or "").upper()
        return f"Add {wname}" + (f" ({wpos})" if wpos else "")
    mu = matchup_one_liner(matchup)
    if mu:
        return mu
    bname = str(breakout_name or "").strip()
    if bname:
        return f"Breakout: {bname}"
    return ""


def leagues_snapshot_table_html(entries: list) -> str:
    """One card: every connected league as a row (name, record, one focus)."""
    rows = [e for e in (entries or []) if isinstance(e, dict) and str(e.get("name") or "").strip()]
    if not rows:
        return ""

    body = ""
    for i, ent in enumerate(rows):
        name = str(ent.get("name") or "").strip()
        href = str(ent.get("href") or "").strip()
        chip = str(ent.get("chip") or "").strip()
        standing = str(ent.get("standing") or "").strip()
        focus = str(ent.get("focus") or "").strip()
        urgent = bool(ent.get("urgent"))
        label = escape(name, quote=False)
        if href:
            label = (
                f'<a class="em-t" href="{escape(href, quote=True)}" style="color:#0f172a;'
                f'text-decoration:none;">{label}</a>'
            )
        meta = escape(chip, quote=False)
        focus_cls = "em-urgent" if urgent else "em-t2"
        focus_color = "#1d4ed8" if urgent else "#334155"
        rowb_cls = "em-rowb" if i else ""
        border = "border-top:1px solid #eef2f7;" if i else ""
        standing_html = (
            f'<td class="{rowb_cls} em-t" style="padding:12px 0 12px 12px;{border}font-size:13px;font-weight:700;'
            f'color:#0f172a;text-align:right;white-space:nowrap;vertical-align:top;">'
            f"{escape(standing, quote=False)}</td>"
            if standing else
            f'<td class="{rowb_cls}" style="padding:12px 0;{border}"></td>'
        )
        focus_html = (
            f'<div class="{focus_cls}" style="font-size:13px;color:{focus_color};margin-top:4px;line-height:1.4;">'
            f"{escape(focus, quote=False)}</div>"
            if focus else ""
        )
        meta_html = (
            f'<div class="em-t3" style="font-size:12px;color:#64748b;margin-top:2px;">{meta}</div>'
            if meta else ""
        )
        body += (
            f"<tr>"
            f'<td class="{rowb_cls}" style="padding:12px 0;{border}vertical-align:top;">'
            f'<div class="em-t" style="font-size:15px;font-weight:700;color:#0f172a;line-height:1.3;">{label}</div>'
            f"{meta_html}{focus_html}</td>"
            f"{standing_html}</tr>"
        )
    return (
        f'<div class="em-sect" style="{EMAIL_CARD_STYLE}">'
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        f'style="width:100%;border-collapse:collapse;">{body}</table></div>'
    )


def league_overview_card_html(
    *,
    league_name: str,
    format_label: str = "",
    rank: Optional[int] = None,
    wins: int = 0,
    losses: int = 0,
    dash_url: str = "",
    matchup: Optional[dict] = None,
    lineup_note: Optional[dict] = None,
    waiver: Optional[dict] = None,
    injury_body: str = "",
    top_asset: Optional[dict] = None,
    riser_name: str = "",
    riser_delta: Optional[float] = None,
    breakout_name: str = "",
    trade_body: str = "",
    is_dynasty: bool = False,
) -> str:
    """Compact one-league row used in snapshot tables and leftover footers."""
    name = str(league_name or "").strip()
    if not name:
        return ""
    games = int(wins or 0) + int(losses or 0)
    standing = ""
    if rank is not None and games > 0:
        standing = f"#{int(rank)} · {int(wins or 0)}-{int(losses or 0)}"
    note = lineup_note or {}
    urgent = bool(str(note.get("body") or note.get("title") or "").strip() or str(injury_body or "").strip())
    focus = league_focus_line(
        is_dynasty=is_dynasty,
        matchup=matchup,
        lineup_note=lineup_note,
        waiver=waiver,
        injury_body=injury_body,
        top_asset=top_asset,
        riser_name=riser_name,
        riser_delta=riser_delta,
        breakout_name=breakout_name,
    )
    return leagues_snapshot_table_html([{
        "name": name,
        "href": dash_url,
        "chip": format_label,
        "standing": standing,
        "focus": focus,
        "urgent": urgent,
    }])


def thursday_alert_html(items: Optional[list], *, compact: bool = False) -> str:
    """Amber alert for starters playing Thursday night. Empty when no items.

    Each item: {name, team, kickoff, league?}. ``compact`` renders a single
    line for the multi-league overview instead of the full alert card.
    """
    rows = [it for it in (items or []) if isinstance(it, dict) and it.get("name")]
    if not rows:
        return ""
    if compact:
        names = ", ".join(
            escape(str(it.get("name") or ""), quote=False) for it in rows[:3]
        )
        return (
            '<p class="em-alert-t" style="margin:0 0 8px;font-size:13px;color:#92400e;line-height:1.5;">'
            f"<strong>Thursday night:</strong> set your lineup early, {names} "
            "play Thursday.</p>"
        )
    parts = []
    kickoffs = {str(it.get("kickoff") or "") for it in rows} - {""}
    for it in rows[:4]:
        nm = escape(str(it.get("name") or ""), quote=False)
        tm = escape(str(it.get("team") or ""), quote=False)
        lg = escape(str(it.get("league") or ""), quote=False)
        label = f"{nm} ({tm})" if tm else nm
        if lg:
            label += f" [{lg}]"
        if len(kickoffs) > 1 and it.get("kickoff"):
            label += f" {escape(str(it.get('kickoff')), quote=False)}"
        parts.append(label)
    body = "Set your lineup before Thursday kickoff: " + ", ".join(parts)
    if len(rows) > 4:
        body += f", and {len(rows) - 4} more"
    if len(kickoffs) == 1:
        body += f". Kickoff {sorted(kickoffs)[0]}"
    body += "."
    return (
        '<div class="em-alert" style="margin:0 0 12px;padding:12px 14px;border-radius:12px;'
        'background:#fffbeb;border:1px solid #fcd34d;">'
        '<div class="em-alert-k" style="font-size:11px;font-weight:800;color:#92400e;'
        'text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;">'
        "Thursday night</div>"
        f'<div class="em-alert-t" style="font-size:14px;color:#78350f;line-height:1.5;">{body}</div>'
        "</div>"
    )


def league_activity_html(bullets: Optional[list], *, href: str = "") -> str:
    """Recent waiver/trade activity around the league. Empty when no bullets."""
    clean = [str(b).strip() for b in (bullets or []) if str(b).strip()]
    if not clean:
        return ""
    items = "".join(
        f'<li class="em-t" style="margin:0 0 6px;font-size:14px;color:#0f172a;line-height:1.5;">'
        f"{escape(b, quote=False)}</li>"
        for b in clean[:4]
    )
    inner = f'<ul style="margin:4px 0 0;padding-left:18px;">{items}</ul>'
    return section_card("Around your league", inner, href=href, cta="Open waivers →")


def league_summary_html(
    *,
    league_name: str,
    rank: Optional[int],
    wins: int = 0,
    losses: int = 0,
    format_label: str = "",
    stakes_line: str = "",
) -> str:
    lg = escape(league_name or "Your league", quote=False)
    games = int(wins or 0) + int(losses or 0)
    if rank and games > 0:
        rec = f"{int(wins or 0)}-{int(losses or 0)}"
        headline = (
            f'You\'re <strong>#{int(rank)}</strong> in {lg} at '
            f'<strong>{escape(rec, quote=False)}</strong>.'
        )
        size = "16px"
    else:
        headline = (
            f'<strong style="font-size:20px;letter-spacing:-0.02em;">{lg}</strong>'
        )
        size = "16px"
    chip = format_chip_html(format_label)
    stakes = ""
    if (stakes_line or "").strip():
        stakes = (
            '<div class="em-t2" style="font-size:13px;color:#475569;margin-top:3px;">'
            f"{escape(stakes_line.strip(), quote=False)}</div>"
        )
    return (
        f'<div class="em-t" style="margin:0 0 8px;font-size:{size};color:#0f172a;line-height:1.4;">'
        f"{headline}{chip}{stakes}</div>"
    )


def matchup_html(matchup: Optional[dict], *, href: str = "") -> str:
    if not matchup:
        return ""
    opp = str(matchup.get("opponent_name") or "").strip()
    if not opp:
        return ""
    you = matchup.get("user_proj")
    them = matchup.get("opp_proj")
    margin = matchup.get("margin")
    wp = matchup.get("win_prob")
    lines = [f"vs {escape(opp)}"]
    if you is not None and them is not None:
        try:
            yu, ot = float(you), float(them)
            lines.append(f"Projected {yu:.1f} to {ot:.1f}")
            if margin is not None:
                m = float(margin)
                if abs(m) >= 0.05:
                    verb = "Favored by" if m > 0 else "Projected behind by"
                    lines.append(f"{verb} {abs(m):.1f}")
        except (TypeError, ValueError):
            pass
    if wp is not None:
        try:
            pct = int(round(float(wp) * 100))
            pct = max(1, min(99, pct))
            lines.append(f"Win probability {pct}%")
        except (TypeError, ValueError):
            pass
    body = ". ".join(lines) + "."
    return action_section_html("This week's matchup", body, href=href, cta="Open matchup →")


def start_sit_html(note: Optional[dict], *, href: str = "") -> str:
    if not note:
        return ""
    title = str(note.get("title") or "Start/Sit")
    body = str(note.get("body") or "")
    if not body:
        return ""
    return action_section_html(title, body, href=href, cta="Fix lineup →")


def waiver_html(
    targets: list,
    *,
    href: str = "",
    base: str = "",
    platform: str = "",
    season: int = 0,
    league_id: str = "",
) -> str:
    if not targets:
        return ""
    rows = ""
    shown = 0
    for t in targets[:3]:
        name = str(t.get("name") or "").strip()
        if not name:
            continue
        pos = str(t.get("pos") or "").upper()
        reason = str(t.get("reason") or "").strip()
        pid = str(t.get("player_id") or "")
        label = escape(name, quote=False)
        if base and platform and season and league_id and pid:
            link = player_deep_link(base, platform, season, league_id, pid, name)
            label = (
                f'<a class="em-link" href="{escape(link, quote=True)}" style="color:#0f172a;'
                f'text-decoration:none;font-weight:700;">{label}</a>'
            )
        meta = " · ".join(p for p in (pos, reason) if p)
        border = "border-top:1px solid #e2e8f0;" if shown else ""
        rowb = "em-rowb" if shown else ""
        rows += (
            f'<tr><td class="{rowb} em-t" style="padding:8px 0;{border}font-size:15px;color:#0f172a;">{label}'
            f'<div class="em-t3" style="font-size:12px;color:#64748b;margin-top:2px;">{escape(meta, quote=False)}</div>'
            f"</td></tr>"
        )
        shown += 1
    if not shown:
        return ""
    title = "Top waiver target" if shown == 1 else "Waiver wire"
    return section_card(
        title,
        f'<table style="width:100%;border-collapse:collapse;">{rows}</table>',
        href=href,
        cta="View waivers →" if href else "",
        accent=True,
    )


def roster_core_html(
    players: list,
    *,
    base: str = "",
    platform: str = "",
    season: int = 0,
    league_id: str = "",
) -> str:
    if not players or len(players) < 2:
        return ""
    rows = ""
    for p in players[:3]:
        name = str(p.get("name") or "").strip()
        if not name:
            continue
        pos = str(p.get("pos") or "").upper()
        try:
            val = float(p.get("value") or 0)
        except (TypeError, ValueError):
            val = 0.0
        pid = str(p.get("player_id") or "")
        label = escape(name, quote=False)
        if base and platform and season and league_id and pid:
            link = player_deep_link(base, platform, season, league_id, pid, name)
            label = (
                f'<a class="em-link" href="{escape(link, quote=True)}" style="color:#0f172a;'
                f'text-decoration:none;font-weight:600;">{label}</a>'
            )
        pos_s = escape(pos, quote=False)
        rows += (
            f'<tr><td class="em-t" style="padding:6px 0;font-size:14px;">{label}'
            f'<div class="em-t3" style="font-size:12px;color:#64748b;">{pos_s}</div></td>'
            f'<td class="em-t" style="padding:6px 0;font-size:14px;font-weight:700;color:#0f172a;'
            f'text-align:right;white-space:nowrap;">{val:.0f}</td></tr>'
        )
    if not rows:
        return ""
    return section_card(
        "Your top assets",
        f'<table style="width:100%;border-collapse:collapse;">{rows}</table>',
        accent=False,
    )


def injury_html(note: Optional[dict], *, href: str = "") -> str:
    if not note:
        return ""
    body = str(note.get("body") or "").strip()
    if not body:
        return ""
    title = str(note.get("title") or "Injury")
    return action_section_html(title, body, href=href, cta="Review roster →")


def _mover_rows(
    pairs: list,
    *,
    up: bool,
    base: str,
    platform: str,
    season: int,
    league_id: str,
    pidx: dict,
    notes: Optional[dict] = None,
) -> str:
    color = "#16a34a" if up else "#dc2626"
    dcls = "em-up" if up else "em-dn"
    arrow = "▲" if up else "▼"
    cells = ""
    for item in pairs:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            pid, d = str(item[0]), item[1]
            extra = item[2] if len(item) > 2 else ""
        elif isinstance(item, dict):
            pid = str(item.get("player_id") or "")
            d = item.get("delta")
            extra = item.get("note") or ""
        else:
            continue
        if not pid or d is None:
            continue
        raw_name = _player_name(pid, pidx)
        if not raw_name:
            continue
        try:
            delta = float(d)
        except (TypeError, ValueError):
            continue
        note = extra or (notes or {}).get(pid) or ""
        nm = escape(raw_name, quote=False)
        href = escape(player_deep_link(base, platform, season, league_id, pid, raw_name), quote=True)
        note_html = (
            f'<div class="em-t3" style="font-size:12px;color:#64748b;font-weight:400;">{escape(str(note), quote=False)}</div>'
            if note else ""
        )
        cells += (
            f'<tr><td class="em-t" style="padding:6px 0;font-size:14px;">'
            f'<a class="em-link" href="{href}" style="color:#0f172a;text-decoration:none;font-weight:600;">'
            f"{nm}</a>{note_html}</td>"
            f'<td class="{dcls}" style="padding:6px 0;font-size:14px;font-weight:700;color:{color};'
            f'text-align:right;white-space:nowrap;">{arrow} {abs(delta):.0f}</td></tr>'
        )
    if not cells:
        return ""
    return f'<table style="width:100%;border-collapse:collapse;">{cells}</table>'


def player_movement_html(
    *,
    my_risers: list,
    my_fallers: list,
    lg_risers: list,
    base: str,
    platform: str,
    season: int,
    league_id: str,
    pidx: dict,
    notes: Optional[dict] = None,
    show_leaguewide: bool = True,
    dynasty: bool = True,
) -> str:
    parts: list[str] = []
    if my_risers:
        title = "Your risers this week" if dynasty else "Roster trends this week"
        table = _mover_rows(
            my_risers, up=True, base=base, platform=platform, season=season,
            league_id=league_id, pidx=pidx, notes=notes,
        )
        if table:
            parts.append(section_card(title, table, accent=False))
    if my_fallers:
        table = _mover_rows(
            my_fallers, up=False, base=base, platform=platform, season=season,
            league_id=league_id, pidx=pidx, notes=notes,
        )
        if table:
            parts.append(section_card("Your fallers this week", table, accent=False))
    if show_leaguewide and lg_risers:
        table = _mover_rows(
            lg_risers, up=True, base=base, platform=platform, season=season,
            league_id=league_id, pidx=pidx, notes=notes,
        )
        if table:
            parts.append(section_card("Biggest risers leaguewide", table, accent=False))
    return "".join(parts)


def breakout_html(watch: Optional[dict], *, href: str = "") -> str:
    if not watch:
        return ""
    name = str(watch.get("name") or "").strip()
    if not name:
        return ""
    score = watch.get("score")
    hit = watch.get("hit_probability")
    bits = [escape(name)]
    try:
        if score is not None:
            bits.append(f"Breakout Score {int(round(float(score)))}")
    except (TypeError, ValueError):
        pass
    try:
        if hit is not None:
            pct = float(hit)
            if pct <= 1:
                pct *= 100
            bits.append(f"{int(round(pct))}% hit rate")
    except (TypeError, ValueError):
        pass
    body = " · ".join(bits)
    return action_section_html("Breakout Watch", body, href=href, cta="Open player →")


def trade_insight_html(insight: Optional[dict], *, href: str = "") -> str:
    if not insight:
        return ""
    body = str(insight.get("body") or "").strip()
    if not body:
        return ""
    title = str(insight.get("title") or "Roster construction")
    return action_section_html(title, body, href=href, cta="Open trades →")


def format_chip(fmt: dict) -> str:
    """Short human label like ``SF · TEP · Dynasty`` without internal ids."""
    if not fmt:
        return ""
    kind = str(fmt.get("type") or "").strip()
    if kind:
        kind = kind[0].upper() + kind[1:]
    parts = ["SF" if fmt.get("is_superflex") else "1QB"]
    if fmt.get("is_tep"):
        parts.append("TEP")
    if kind:
        parts.append(kind)
    return " · ".join(parts)


def _player_name(pid: str, pidx: dict) -> str:
    meta = (pidx or {}).get(str(pid)) or {}
    name = (
        meta.get("full_name")
        or meta.get("name")
        or ((meta.get("first_name") or "") + " " + (meta.get("last_name") or "")).strip()
    )
    name = str(name or "").strip()
    if not name or name == str(pid) or name.lower().startswith("player "):
        return ""
    return name


# ======================================================================
# From utils/cross_league_actions.py
# ======================================================================

"""Cross-league action digest helpers (roadmap R04).

Ranks per-league to-dos for the My Leagues hub. Pure functions so ranking is
unit-testable without Flask / provider I/O.
"""

import re

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
    ir_used: Optional[int] = None,
    ir_slots: Optional[int] = None,
) -> Optional[dict]:
    v = str(verdict or "").strip()
    if v not in ("IR", "Move to IR", "Drop candidate", "Stash"):
        return None
    # Already occupying an IR slot — stash/move-to-IR tips are not actionable.
    # Drop candidate can still matter (free the IR slot).
    if already_on_ir and v in ("IR", "Move to IR", "Stash"):
        return None
    move = v in ("IR", "Move to IR")
    if move and (ir_slots is None or ir_used is None or ir_slots <= ir_used):
        return None
    title = f"Move {player_name} to IR" if move else f"{v}: {player_name}"
    detail = f"Approx return {weeks_label}" if weeks_label else "Approximate injury guidance"
    if move:
        detail = f"{ir_used} of {ir_slots} IR slots used · {detail}"
    return make_action(
        kind="injury",
        platform=platform,
        season=season,
        league_id=league_id,
        league_name=league_name,
        title=title,
        detail=detail,
        href=f"/{(platform or 'sleeper').strip().lower()}/{int(season)}/{league_id}/waivers?tab=startsit",
        severity=0.8 if v == "Drop candidate" else 0.55,
    )


# Waiver pickups are only worth a cross-league nudge when the add is actually
# startable (or fills a hole). Highest leftover value is not enough — RB35 /
# QB21 / WR44 are the top *remaining* names in most leagues, not great adds.
#
# Value scale is the shared model (WEIGHTS.min_value == 25). Redraft churns the
# wire for rest-of-season production (lower bar); dynasty only pings for a
# genuine long-term asset (higher bar). Rank ceilings are 12-team startable
# depth, scaled by league size.
WAIVER_MIN_VALUE_MULT = {"redraft": 1.6, "dynasty": 2.4}

# Positional ranks at/inside this are weekly-startable in a typical 12-team
# league. Deeper than this is replacement-level leftover, not a "this week's
# move" unless the roster has a real starter hole (short stretch below).
WAIVER_RANK_CEILING_12 = {
    "redraft": {"QB": 14, "RB": 28, "WR": 32, "TE": 12},
    "dynasty": {"QB": 18, "RB": 30, "WR": 40, "TE": 14},
}
WAIVER_SF_QB_CEILING_12 = {"redraft": 24, "dynasty": 28}

# Aging dynasty players must still be clearly startable — WR35 Tyreek is not a
# stash just because he is the highest leftover dynasty value.
WAIVER_VETERAN_RANK_CEILING_12 = {"QB": 12, "RB": 20, "WR": 24, "TE": 10}

_SKILL_POS = ("QB", "RB", "WR", "TE")
_OUT_STATUS = {"IR", "PUP", "NFI", "OUT", "SUSP", "DOUBTFUL"}
_FA_TEAMS = {"", "FA", "FREE AGENT", "N/A"}
_POS_RANK_RE = re.compile(r"(\d+)\s*$")


def waiver_value_threshold(base_min_value: float, *, is_redraft: bool) -> float:
    """Format-aware minimum value for a waiver pickup to be worth surfacing."""
    try:
        base = float(base_min_value)
    except (TypeError, ValueError):
        base = 25.0
    return base * WAIVER_MIN_VALUE_MULT["redraft" if is_redraft else "dynasty"]


def _clamp_league_size(n_teams) -> int:
    try:
        n = int(n_teams or 12)
    except (TypeError, ValueError):
        n = 12
    return max(8, min(16, n))


def waiver_rank_ceiling(
    pos: str,
    *,
    is_redraft: bool,
    is_sf: bool = False,
    n_teams: int = 12,
) -> int:
    """Deepest positional rank that still counts as a worthwhile add."""
    fmt = "redraft" if is_redraft else "dynasty"
    scale = _clamp_league_size(n_teams) / 12.0
    p = str(pos or "").upper()
    if p == "QB" and is_sf:
        base = WAIVER_SF_QB_CEILING_12[fmt]
    else:
        base = WAIVER_RANK_CEILING_12[fmt].get(p, 24)
    return max(8, int(round(base * scale)))


def parse_pos_rank(rank=None, label: str = "") -> Optional[int]:
    """Numeric positional rank from a value-table field or 'RB35' label."""
    try:
        if rank not in (None, ""):
            n = int(rank)
            if n > 0:
                return n
    except (TypeError, ValueError):
        pass
    m = _POS_RANK_RE.search(str(label or ""))
    return int(m.group(1)) if m else None


def waiver_add_clears_quality_bar(
    *,
    pos: str,
    pos_rank: Optional[int],
    value: float,
    is_redraft: bool,
    is_sf: bool = False,
    n_teams: int = 12,
    age: float = 0.0,
    starter_gap: float = 0.0,
    min_value: float = 25.0,
) -> bool:
    """True when this free agent is worth a cross-league 'Add' nudge.

    Highest remaining value is not enough. The player must clear the format
    value floor *and* be startable-ish (or fill a real starter hole). 1QB
    streamers and past-prime dynasty leftovers are dropped even when they are
    the top name on a barren wire.
    """
    try:
        val = float(value or 0)
    except (TypeError, ValueError):
        val = 0.0
    threshold = waiver_value_threshold(min_value, is_redraft=is_redraft)
    if val < threshold:
        return False

    p = str(pos or "").upper()
    if p not in _SKILL_POS:
        return False

    try:
        gap = float(starter_gap or 0)
    except (TypeError, ValueError):
        gap = 0.0

    ceiling = waiver_rank_ceiling(
        p, is_redraft=is_redraft, is_sf=is_sf, n_teams=n_teams,
    )
    # A lineup hole can stretch slightly past the startable line; replacement
    # RBs (RB35+) still do not qualify.
    stretch = int(round(ceiling * 1.15)) if gap > 0 else ceiling

    if pos_rank is None:
        # Unknown rank: only ping when the value is unambiguously real.
        return val >= threshold * 2.0

    try:
        rk = int(pos_rank)
    except (TypeError, ValueError):
        return False
    if rk <= 0 or rk > stretch:
        return False

    # 1QB: QB15-24 are streamers, not weekly adds, unless the roster cannot
    # start a quarterback.
    if p == "QB" and not is_sf:
        qb_bar = max(8, int(round(12 * (_clamp_league_size(n_teams) / 12.0))))
        if gap <= 0 and rk > qb_bar:
            return False

    # Dynasty: aging players must still be clearly startable.
    if not is_redraft:
        try:
            age_f = float(age or 0)
        except (TypeError, ValueError):
            age_f = 0.0
        if age_f:
            from utils.waivers import WAIVER_PRIME_MAX
            prime = float(WAIVER_PRIME_MAX.get(p, 28))
            if age_f > prime + 1:
                vet = max(8, int(round(
                    WAIVER_VETERAN_RANK_CEILING_12.get(p, 16)
                    * (_clamp_league_size(n_teams) / 12.0)
                )))
                if rk > vet:
                    return False
    return True


def waiver_add_detail(
    *,
    pos_rank_label: str = "",
    position: str = "",
    is_redraft: bool = False,
    starter_gap: float = 0.0,
    pos_rank: Optional[int] = None,
    is_sf: bool = False,
    n_teams: int = 12,
) -> str:
    """One-line reason for a cross-league waiver card (not 'top leftover')."""
    bits = []
    if pos_rank_label:
        bits.append(str(pos_rank_label).strip())
    pos = str(position or "").upper()
    if starter_gap and starter_gap > 0 and pos:
        bits.append(f"Fills a {pos} need")
    else:
        startable = waiver_rank_ceiling(
            pos or "WR", is_redraft=is_redraft, is_sf=is_sf, n_teams=n_teams,
        )
        # Tighter than the surfacing ceiling: "startable" copy is for names that
        # would actually be in a lineup, not the last name that cleared the bar.
        startable = max(8, int(round(startable * 0.85)))
        if pos_rank and pos_rank <= startable:
            bits.append("Startable on the wire")
        else:
            fmt = "redraft" if is_redraft else "dynasty"
            bits.append(f"Worth adding by {fmt} value")
    return " · ".join(bits)


def _waiver_need_context(
    roster_players: Optional[list],
    roster_positions: Optional[list],
    pidx: Optional[dict],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (need_mult by pos, starter_gap by pos) for the viewer's roster."""
    need_mults = {p: 1.0 for p in _SKILL_POS}
    gaps = {p: 0.0 for p in _SKILL_POS}
    try:
        from utils.lineups import count_lineup_slots, start_sit_pos, starter_need_counts
        from utils.waivers import need_multiplier, positional_need_scores
        counts: dict[str, int] = {}
        for pid in roster_players or []:
            meta = (pidx or {}).get(str(pid)) or {}
            pos = start_sit_pos(meta.get("position") or meta.get("pos") or "")
            if pos in _SKILL_POS:
                counts[pos] = counts.get(pos, 0) + 1
        targets = starter_need_counts(roster_positions or [], extra_depth=1)
        slots = count_lineup_slots(roster_positions or [])
        # A "hole" is fewer bodies than dedicated starters — FLEX does not
        # create a WR/RB need by itself, or every 2-WR league looks empty.
        dedicated = {
            "QB": max(1, int(slots.get("QB") or 0) + int(slots.get("SUPER_FLEX") or 0)),
            "RB": max(1, int(slots.get("RB") or 0)),
            "WR": max(1, int(slots.get("WR") or 0)),
            "TE": max(1, int(slots.get("TE") or 0)),
        }
        need_scores = positional_need_scores(counts, targets)
        need_mults = {pos: need_multiplier(pos, need_scores) for pos in _SKILL_POS}
        for pos in _SKILL_POS:
            gaps[pos] = max(0.0, float(dedicated.get(pos, 0)) - float(counts.get(pos, 0)))
    except Exception:
        pass
    return need_mults, gaps


def select_waiver_add(
    rows: list,
    rostered_ids: set[str],
    *,
    value_key: str,
    fallback_key: str = "value",
    rank_key: str = "pos_rank",
    rank_label_key: str = "pos_rank_label",
    is_redraft: bool = False,
    is_sf: bool = False,
    n_teams: int = 12,
    roster_players: Optional[list] = None,
    roster_positions: Optional[list] = None,
    pidx: Optional[dict] = None,
    min_value: float = 25.0,
    injured_status_by_pid: Optional[dict] = None,
) -> Optional[dict]:
    """Pick one actually-worth-adding free agent, or None.

    Ranks eligible names with ``waiver_pickup_score`` (value, age, roster need,
    trend) after the quality bar drops leftover RB35s / streamer QBs / aging
    dynasty depth. Empty when the wire has nobody worth a cross-league nudge.
    """
    from utils.waivers import waiver_pickup_score

    owned = {str(p) for p in (rostered_ids or set())}
    pidx = pidx or {}
    inj_map = injured_status_by_pid or {}
    need_mults, gaps = _waiver_need_context(roster_players, roster_positions, pidx)

    best: Optional[dict] = None
    best_score = -1.0
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("id") or row.get("player_id") or "").strip()
        if not pid or pid in owned:
            continue
        pos = str(row.get("position") or row.get("pos") or "").upper()
        if pos not in _SKILL_POS:
            continue
        meta = pidx.get(pid) or {}
        team = str(row.get("team") or meta.get("team") or "").upper()
        if team in _FA_TEAMS:
            continue
        inj = str(inj_map.get(pid) or meta.get("injury_status") or "").upper()
        if inj in _OUT_STATUS:
            continue
        try:
            val = float(row.get(value_key) or row.get(fallback_key) or row.get("value") or 0.0)
        except (TypeError, ValueError):
            val = 0.0
        label = str(row.get(rank_label_key) or row.get("pos_rank_label") or "")
        rk = parse_pos_rank(row.get(rank_key) or row.get("pos_rank"), label)
        try:
            age = float(row.get("age") or meta.get("age") or 0)
        except (TypeError, ValueError):
            age = 0.0
        gap = gaps.get(pos, 0.0)
        if not waiver_add_clears_quality_bar(
            pos=pos, pos_rank=rk, value=val, is_redraft=is_redraft,
            is_sf=is_sf, n_teams=n_teams, age=age, starter_gap=gap,
            min_value=min_value,
        ):
            continue
        name = (
            row.get("name") or row.get("full_name") or row.get("player")
            or meta.get("name") or meta.get("full_name") or ""
        )
        name = str(name).strip()
        if not name or name == pid:
            continue
        try:
            chg = float(row.get("rank_change_7d") or 0)
        except (TypeError, ValueError):
            chg = 0.0
        cand = {
            "player_id": pid,
            "value": val,
            "age": age,
            "position": pos,
            "rank_change_7d": chg,
            "need_mult": need_mults.get(pos, 1.0),
            "pos_rank": rk,
            "self_status": inj,
        }
        try:
            score = float(waiver_pickup_score(cand, {}))
        except Exception:
            score = val
        if score <= best_score:
            continue
        best_score = score
        rank_t = 0.0 if not rk else max(0.0, 1.0 - (rk / 40.0))
        need_t = min(1.0, max(0.0, gap) / 2.0)
        severity = 0.35 + 0.35 * rank_t + 0.30 * need_t
        best = {
            "player_id": pid,
            "name": name,
            "position": pos,
            "pos_rank_label": label,
            "pos_rank": rk,
            "value": val,
            "score": score,
            "starter_gap": gap,
            "severity": severity,
            "reason": waiver_add_detail(
                pos_rank_label=label, position=pos, is_redraft=is_redraft,
                starter_gap=gap, pos_rank=rk, is_sf=is_sf, n_teams=n_teams,
            ),
        }
    return best


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
    reason: str = "",
    severity: float = 0.0,
    starter_gap: float = 0.0,
    pos_rank: Optional[int] = None,
    is_sf: bool = False,
    n_teams: int = 12,
) -> dict:
    """A waiver add that already cleared ``select_waiver_add``'s quality bar."""
    pos = str(position or "").upper()
    name = str(player_name or "").strip() or "a free agent"
    title = f"Add {name}" + (f" ({pos})" if pos else "")
    detail = str(reason or "").strip() or waiver_add_detail(
        pos_rank_label=pos_rank_label, position=pos, is_redraft=is_redraft,
        starter_gap=starter_gap, pos_rank=pos_rank, is_sf=is_sf, n_teams=n_teams,
    )
    try:
        sev = float(severity or 0)
    except (TypeError, ValueError):
        sev = 0.0
    if sev <= 0:
        sev = 0.5
    return make_action(
        kind="waiver",
        platform=platform,
        season=season,
        league_id=league_id,
        league_name=league_name,
        title=title,
        detail=detail,
        severity=sev,
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


# ======================================================================
# From utils/injury_plan.py
# ======================================================================

"""Approximate injury return planner (roadmap R07).

Never claims medical certainty. ESPN return dates are preferred; otherwise we
fall back to status-class duration bands from the waiver model.
"""

from utils.waivers import INJURY_DURATION_WEEKS

# Dynasty value thresholds for stash vs drop (display values, not SF).
_VALUE_STASH = 80.0
_VALUE_HOLD = 40.0


def status_weeks_band(status: Optional[str]) -> Optional[float]:
    """Fixed duration band for a Sleeper/ESPN injury status class."""
    st = str(status or "").strip().upper()
    if not st or st in ("ACTIVE", "ACT", "HEALTHY", ""):
        return None
    # Common aliases
    if st in ("O",):
        st = "OUT"
    if st in ("D",):
        st = "DOUBTFUL"
    if st in ("Q", "GTD"):
        st = "QUESTIONABLE"
    if st in ("SUS", "SUSPENDED"):
        st = "SUSP"
    return INJURY_DURATION_WEEKS.get(st)


def resolve_weeks_out(
    *,
    status: Optional[str] = None,
    espn_weeks: Optional[float] = None,
) -> tuple[Optional[float], str]:
    """Return (weeks, source) with source ``espn`` | ``status`` | ``none``."""
    try:
        if espn_weeks is not None and float(espn_weeks) >= 0:
            return float(espn_weeks), "espn"
    except (TypeError, ValueError):
        pass
    band = status_weeks_band(status)
    if band is not None:
        return float(band), "status"
    return None, "none"


def injury_plan(
    *,
    status: Optional[str] = None,
    espn_weeks: Optional[float] = None,
    player_value: Optional[float] = None,
    has_open_ir_slot: bool = False,
    already_on_ir: bool = False,
) -> Optional[dict[str, Any]]:
    """Heuristic stash / drop / IR verdict for an injured player.

    Returns ``None`` when the player is not injured. Always sets
    ``approximate: True`` and hedging copy in ``reason``.
    """
    st = str(status or "").strip()
    weeks, source = resolve_weeks_out(status=st, espn_weeks=espn_weeks)
    if weeks is None and not st:
        return None
    if not st and weeks is None:
        return None
    # Healthy / no meaningful designation
    if st and st.upper() in ("ACTIVE", "ACT", "HEALTHY") and weeks is None:
        return None

    try:
        val = float(player_value) if player_value is not None else None
    except (TypeError, ValueError):
        val = None

    if already_on_ir:
        verdict = "Already on IR"
        reason = "Player is already in a reserve slot; no roster move is required."
    elif weeks is not None and weeks <= 1.0:
        verdict = "Monitor"
        reason = "Listed return is soon (approx). Check inactive reports before lock."
    elif weeks is not None and weeks <= 3.0:
        if val is not None and val >= _VALUE_HOLD:
            verdict = "Hold"
            reason = "Short absence (~%.0f wk approx) and enough roster value to hold." % weeks
        else:
            verdict = "Drop candidate"
            reason = "Short absence but limited stash value. Free the spot if you need it."
    elif weeks is not None and weeks <= 6.0:
        if has_open_ir_slot:
            verdict = "Move to IR"
            reason = "Multi-week absence (~%.0f wk approx). Use an IR slot if available." % weeks
        elif val is not None and val >= _VALUE_STASH:
            verdict = "Hold"
            reason = "Longer absence (~%.0f wk approx) but high value. Stash if you can." % weeks
        else:
            verdict = "Drop candidate"
            reason = "Longer absence (~%.0f wk approx) without IR room. Lean drop unless deep bench." % weeks
    else:
        # IR / PUP / unknown long
        if has_open_ir_slot or (st and st.upper() in ("IR", "PUP", "NFI")):
            verdict = "Move to IR"
            reason = "Extended absence (approx). IR if the league allows; otherwise stash only if elite."
        elif val is not None and val >= _VALUE_STASH:
            verdict = "Hold"
            reason = "Extended absence (approx) but elite value. Hold through the window if possible."
        else:
            verdict = "Drop candidate"
            reason = "Extended absence (approx) with limited stash value."

    if weeks is not None and weeks < 1:
        weeks_label = "~this week"
    elif weeks is not None:
        weeks_label = f"~{weeks:.0f} wk" if weeks >= 1 else f"~{weeks:.1f} wk"
    else:
        weeks_label = "unknown window"

    return {
        "verdict": verdict,
        "reason": reason,
        "weeks_out": weeks,
        "weeks_label": weeks_label,
        "source": source,
        "approximate": True,
        "status": st or None,
    }


def ir_capacity(
    roster_positions: Optional[Sequence] = None,
    reserve_ids: Optional[Sequence] = None,
    reserve_slots: Optional[int] = None,
) -> dict[str, Any]:
    """How many IR slots the league has and whether one is open."""
    from utils.lineups import canonicalize_slots

    slots = canonicalize_slots(roster_positions or [])
    fallback = sum(1 for s in slots if s == "IR")
    try:
        ir_slots = max(0, int(reserve_slots)) if reserve_slots is not None else fallback
    except (TypeError, ValueError):
        ir_slots = fallback
    # Providers use IR, IR+, RES, RESERVE and injured_reserve interchangeably;
    # adapters pass their contents here, so only nonblank canonical IDs count.
    used = len({str(x).strip() for x in (reserve_ids or []) if str(x or "").strip()})
    open_count = max(0, ir_slots - used)
    return {
        "has_ir_slot": ir_slots > 0,
        "has_open_ir_slot": open_count > 0,
        "ir_open": open_count > 0,  # backwards-compatible key
        "ir_open_count": open_count,
        "ir_slots": ir_slots,
        "ir_used": used,
    }
