"""Shared data for one weekly-digest run.

Loads recipient-independent datasets once and caches per-league payloads so
users in the same league do not refetch. A failure loading one league does not
abort the run.
"""
from __future__ import annotations

import logging
from math import erf, sqrt
from typing import Any, Optional

logger = logging.getLogger(__name__)

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
            from utils.utils import load_players_index
            self.pidx = load_players_index() or {}
            self.movers_1qb = get_top_movers(days=7, limit=2000, league_type="1qb") or {}
            self.movers_sf = get_top_movers(days=7, limit=2000, league_type="sf") or {}
        except Exception:
            logger.debug("[digest-cache] movers/index load failed", exc_info=True)
            self.pidx = self.pidx or {}
            self.movers_1qb = self.movers_1qb or {}
            self.movers_sf = self.movers_sf or {}
        try:
            from utils.utils import load_model_value_table
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
            from utils.utils import load_week_schedule
            for g in load_week_schedule(season, week) or []:
                for side in ("home", "away"):
                    t = str(g.get(side) or "").upper()
                    if t:
                        self.teams_playing.add(t)
        except Exception:
            logger.debug("[digest-cache] schedule load failed", exc_info=True)
        try:
            from utils.utils import load_week_projection
            from utils.fantasy_scoring import weekly_projection_points
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
    from utils.league_format import classify_league_roster_format

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
        from utils.lineup_slots import count_lineup_slots
        from utils.roster_strength import weighted_pos_strength
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
