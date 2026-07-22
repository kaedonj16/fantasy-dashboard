"""Keeper Assistant page.

Server-renders the "who should I keep?" tool for a league: an optimizer that
picks the best keepers under the league limit, with a full sortable table
available as a toggle. All the decision math lives in utils.keeper_value (pure +
unit-tested); this module just assembles real roster / draft / ADP / value data
into candidates and hands them to the client, which re-runs the same math live
as the manager tweaks the keeper limit and cost rules.

Draft-round auto-detection uses Sleeper's draft results; on ESPN/Yahoo (and for
waiver adds anywhere) the round starts blank and the manager sets it; the tool
still works, it just can't pre-fill the cost.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from utils.keeper_value import (
    KeeperRules, KeeperCandidate, evaluate, analyze, project_league_keepers,
)

logger = logging.getLogger(__name__)

_VERDICT_LABEL = {"keep": "KEEP", "toss": "TOSS-UP", "pass": "PASS"}


def _is_superflex(ctx: Dict[str, Any]) -> bool:
    positions = ctx.get("roster_positions") or ctx.get("roster_positions_list") or []
    return any(str(p).upper() in {"SUPER_FLEX", "SFLEX", "SUPERFLEX"} for p in positions)


def _redraft_value_map(is_sf: bool) -> Dict[str, float]:
    """player_id -> redraft value, from the daily player_values table."""
    try:
        from dashboard_services.player_value_history import load_current_values_from_db
        rows = load_current_values_from_db() or []
    except Exception:
        logger.debug("[keeper] value load failed", exc_info=True)
        return {}
    key = "redraft_value_sf" if is_sf else "redraft_value_1qb"
    out: Dict[str, float] = {}
    for r in rows:
        pid = str(r.get("id") or "")
        if not pid:
            continue
        try:
            out[pid] = float(r.get(key) or r.get("redraft_value_1qb") or 0.0)
        except (TypeError, ValueError):
            out[pid] = 0.0
    return out


def _adp_map(is_sf: bool, season: int, source: str = "consensus") -> Dict[str, float]:
    """player_id -> overall redraft ADP via the shared resolver.

    Keeper decisions are redraft, so this always asks the redraft axis. ``source``
    is one of sleeper / yahoo / consensus (for the future source selector);
    Sleeper is the reliable server-reachable feed, Yahoo is redraft-only, and
    consensus blends what's available. Empty result -> caller falls back to the
    value rank."""
    try:
        from dashboard_services.adp_service import resolve_market_adp
        return resolve_market_adp(int(season), is_sf, scoring_type="redraft", source=source) or {}
    except Exception:
        logger.debug("[keeper] adp resolve failed", exc_info=True)
        return {}


def _draft_rounds(d: Optional[Dict[str, Any]]) -> int:
    try:
        return int(((d or {}).get("settings") or {}).get("rounds") or 0)
    except (TypeError, ValueError):
        return 0


def _best_draft(drafts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The draft most likely to hold where current players were acquired: a
    completed draft with the most rounds (a startup / full draft rather than a
    small annual rookie draft), falling back to the most recent listed."""
    pool = [d for d in (drafts or []) if str(d.get("status")) == "complete"] or list(drafts or [])
    if not pool:
        return None
    return max(pool, key=_draft_rounds)


def _drafted_round_map(platform: str, league_id: str) -> Dict[str, int]:
    """player_id -> the round they were drafted (Sleeper only; empty otherwise)."""
    if (platform or "").lower() != "sleeper":
        return {}
    try:
        from dashboard_services.api import get_drafts, get_draft_picks
        draft = _best_draft(get_drafts(league_id) or [])
        if not draft:
            return {}
        picks = get_draft_picks(str(draft.get("draft_id"))) or []
    except Exception:
        logger.debug("[keeper] draft load failed", exc_info=True)
        return {}
    out: Dict[str, int] = {}
    for p in picks:
        pid = str(p.get("player_id") or "")
        rnd = p.get("round")
        if pid and rnd:
            try:
                out[pid] = int(rnd)
            except (TypeError, ValueError):
                continue
    return out


def _num_rounds(platform: str, league_id: str, default: int = 15) -> int:
    """Draft rounds for the keeper-cost scale. Uses the startup/full draft's
    round count; defaults to a standard redraft depth when it can't be detected
    or looks like a small rookie-only draft (which would make the undrafted cost
    absurdly cheap)."""
    if (platform or "").lower() != "sleeper":
        return default
    try:
        from dashboard_services.api import get_drafts
        rounds = _draft_rounds(_best_draft(get_drafts(league_id) or []))
        # A tiny round count is almost certainly a rookie draft, not the main
        # draft the keeper cost should scale against.
        return rounds if rounds >= 8 else default
    except Exception:
        return default


def _detected_keeper_limit(ctx: Dict[str, Any]) -> int:
    """The league's real keeper limit from settings, or 0 if none is configured."""
    for src in (ctx.get("league_settings"), ctx.get("settings"), ctx):
        try:
            v = int((src or {}).get("max_keepers") or 0)
            if v > 0:
                return v
        except (TypeError, ValueError, AttributeError):
            continue
    return 0


def _max_keepers(ctx: Dict[str, Any], default: int = 2) -> int:
    return _detected_keeper_limit(ctx) or default


def league_keeper_limit(ctx: Dict[str, Any]) -> int:
    """Public: real keeper limit for gating (0 = not a detected keeper league).
    The draft room uses this to decide whether to auto-surface keepers."""
    return _detected_keeper_limit(ctx)


def _viewer_roster(ctx: Dict[str, Any], viewer_roster_id: Optional[str]) -> Optional[Dict[str, Any]]:
    rosters = ctx.get("rosters") or []
    if viewer_roster_id:
        for r in rosters:
            if str(r.get("roster_id")) == str(viewer_roster_id):
                return r
    return rosters[0] if rosters else None


def _value_rank_map(values: Dict[str, float]) -> Dict[str, float]:
    """player_id -> overall rank by redraft value (1 = most valuable).

    Used as an ADP proxy when market ADP is unavailable (common in the offseason,
    or when the FantasyCalc fetch is empty) so surplus is still computable instead
    of every player showing "off-board"."""
    ranked = sorted(
        ((pid, v) for pid, v in values.items() if v and v > 0),
        key=lambda kv: -kv[1],
    )
    return {pid: float(i + 1) for i, (pid, _v) in enumerate(ranked)}


def _candidates_for_ids(
    player_ids: List[str],
    players_index: Dict[str, Any],
    values: Dict[str, float],
    adp: Dict[str, float],
    drafted: Dict[str, int],
    value_rank: Optional[Dict[str, float]] = None,
) -> List[KeeperCandidate]:
    """Build keeper candidates for a set of player ids (one team's roster).
    Market ADP falls back to the redraft value rank when it's missing."""
    value_rank = value_rank or {}
    out: List[KeeperCandidate] = []
    for pid in player_ids:
        meta = players_index.get(pid) or {}
        out.append(KeeperCandidate(
            player_id=pid,
            name=meta.get("name") or f"Player {pid}",
            position=(meta.get("pos") or meta.get("position") or "").upper(),
            drafted_round=drafted.get(pid),   # None → UI/user sets it
            years_kept=0,
            adp_overall=adp.get(pid) or value_rank.get(pid),
            value=values.get(pid, 0.0),
        ))
    return out


def compute_league_keepers(
    ctx: Dict[str, Any],
    platform: str = "sleeper",
    league_id: str = "",
    viewer_roster_id: Optional[str] = None,
    viewer_kept_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """League-wide keeper set for the draft board.

    Every team's likely keepers are projected with the surplus optimizer; the
    viewer's team is replaced by their *actual* selections when ``viewer_kept_ids``
    is supplied (the handoff from the keeper page). Returns a payload the draft
    room seeds into __draftCfg.keepers:

        {limit, viewerRoster, autoDraft,
         byTeam: {roster_id: [player_id, ...]},
         kept:   [{id, name, pos, rosterId, costRound, projected}, ...]}
    """
    is_sf = _is_superflex(ctx)
    try:
        league_size = int(ctx.get("total_rosters") or len(ctx.get("rosters") or []) or 12)
    except (TypeError, ValueError):
        league_size = 12
    num_rounds = _num_rounds(platform, league_id)
    limit = _max_keepers(ctx)

    players_index: Dict[str, Any] = {}
    try:
        from utils.utils import load_players_index
        players_index = load_players_index() or {}
    except Exception:
        logger.debug("[keeper] players_index load failed", exc_info=True)
    values = _redraft_value_map(is_sf)
    adp = _adp_map(is_sf, int(ctx.get("season") or 0))
    value_rank = _value_rank_map(values)
    drafted = _drafted_round_map(platform, league_id)
    rules = KeeperRules(league_size=league_size, num_rounds=num_rounds)

    per_team: Dict[str, List[KeeperCandidate]] = {}
    for r in (ctx.get("rosters") or []):
        rid = str(r.get("roster_id"))
        pids = [str(p) for p in (r.get("players") or [])]
        per_team[rid] = _candidates_for_ids(pids, players_index, values, adp, drafted, value_rank)

    by_team = project_league_keepers(per_team, rules, limit)

    vr = str(viewer_roster_id) if viewer_roster_id is not None else None
    if vr is not None and viewer_kept_ids is not None:
        by_team[vr] = [str(x) for x in viewer_kept_ids]

    kept: List[Dict[str, Any]] = []
    for rid, ids in by_team.items():
        cand_by_id = {c.player_id: c for c in per_team.get(rid, [])}
        for pid in ids:
            c = cand_by_id.get(pid)
            if not c:
                continue
            analyze(c, rules)
            kept.append({
                "id": pid, "name": c.name, "pos": c.position,
                "rosterId": rid, "costRound": c.cost_round,
                "projected": not (vr is not None and rid == vr and viewer_kept_ids is not None),
            })
    return {
        "limit": limit, "viewerRoster": vr, "autoDraft": bool(drafted),
        "byTeam": by_team, "kept": kept,
    }


def build_keeper_body(
    ctx: Dict[str, Any],
    viewer_roster_id: Optional[str] = None,
    platform: str = "sleeper",
    league_id: str = "",
) -> str:
    """Return the Keeper Assistant page body HTML for a league context."""
    from dashboard_services.pages._keeper_render import render_keeper_html  # local import: keeps this module import-light

    is_sf = _is_superflex(ctx)
    try:
        league_size = int(ctx.get("total_rosters") or len(ctx.get("rosters") or []) or 12)
    except (TypeError, ValueError):
        league_size = 12
    num_rounds = _num_rounds(platform, league_id)
    max_keepers = _max_keepers(ctx)

    players_index = {}
    try:
        from utils.utils import load_players_index
        players_index = load_players_index() or {}
    except Exception:
        logger.debug("[keeper] players_index load failed", exc_info=True)

    values = _redraft_value_map(is_sf)
    adp = _adp_map(is_sf, int(ctx.get("season") or 0))
    value_rank = _value_rank_map(values)
    drafted = _drafted_round_map(platform, league_id)

    roster = _viewer_roster(ctx, viewer_roster_id) or {}
    player_ids = [str(p) for p in (roster.get("players") or [])]

    candidates = _candidates_for_ids(player_ids, players_index, values, adp, drafted, value_rank)
    rules = KeeperRules(league_size=league_size, num_rounds=num_rounds)
    ranked = evaluate(candidates, rules, limit=max_keepers)

    _plat = (platform or "sleeper").lower()
    _season = ctx.get("season") or ""
    draft_url = (f"/{_plat}/{_season}/{league_id}/draft?keepers=1"
                 if (league_id and _season) else "")
    seed = {
        "leagueSize": league_size,
        "numRounds": num_rounds,
        "maxKeepers": max_keepers,
        "isSuperflex": is_sf,
        "autoDraft": bool(drafted),   # did we auto-detect draft rounds?
        "platform": _plat,
        "leagueId": str(league_id or ""),
        "draftUrl": draft_url,
        "viewerRoster": str(viewer_roster_id) if viewer_roster_id is not None else "",
        "players": [
            {
                "id": c.player_id, "name": c.name, "pos": c.position,
                "draftedRound": c.drafted_round, "yearsKept": c.years_kept,
                "adpOverall": c.adp_overall, "value": round(c.value or 0.0, 1),
            }
            for c in ranked
        ],
    }
    return render_keeper_html(seed)
