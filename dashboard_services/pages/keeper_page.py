"""Keeper Assistant page.

Server-renders the "who should I keep?" tool for a league: an optimizer that
picks the best keepers under the league limit, with a full sortable table
available as a toggle. All the decision math lives in utils.keeper_value (pure +
unit-tested); this module just assembles real roster / draft / ADP / value data
into candidates and hands them to the client, which re-runs the same math live
as the manager tweaks the keeper limit and cost rules.

Draft-round auto-detection uses Sleeper's draft results; on ESPN/Yahoo (and for
waiver adds anywhere) the round starts blank and the manager sets it — the tool
still works, it just can't pre-fill the cost.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from utils.keeper_value import KeeperRules, KeeperCandidate, evaluate

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


def _adp_map(is_sf: bool) -> Dict[str, float]:
    """player_id -> overall redraft ADP (1 = consensus #1 pick)."""
    try:
        from dashboard_services.adp_service import fetch_fc_redraft_adp
        raw = fetch_fc_redraft_adp(is_sf) or {}
    except Exception:
        logger.debug("[keeper] adp load failed", exc_info=True)
        return {}
    out: Dict[str, float] = {}
    for pid, info in raw.items():
        try:
            ov = float((info or {}).get("avg_pick") or (info or {}).get("adp_rank") or 0)
        except (TypeError, ValueError):
            ov = 0.0
        if ov > 0:
            out[str(pid)] = ov
    return out


def _drafted_round_map(platform: str, league_id: str) -> Dict[str, int]:
    """player_id -> the round they were drafted (Sleeper only; empty otherwise)."""
    if (platform or "").lower() != "sleeper":
        return {}
    try:
        from dashboard_services.api import get_drafts, get_draft_picks
        drafts = get_drafts(league_id) or []
        # Prefer a completed draft; fall back to the most recent listed.
        draft = next((d for d in drafts if str(d.get("status")) == "complete"), None) \
            or (drafts[0] if drafts else None)
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
    if (platform or "").lower() != "sleeper":
        return default
    try:
        from dashboard_services.api import get_drafts
        drafts = get_drafts(league_id) or []
        draft = next((d for d in drafts if str(d.get("status")) == "complete"), None) \
            or (drafts[0] if drafts else None)
        rounds = int(((draft or {}).get("settings") or {}).get("rounds") or 0)
        return rounds or default
    except Exception:
        return default


def _max_keepers(ctx: Dict[str, Any], default: int = 2) -> int:
    for src in (ctx.get("league_settings"), ctx.get("settings"), ctx):
        try:
            v = int((src or {}).get("max_keepers") or 0)
            if v > 0:
                return v
        except (TypeError, ValueError, AttributeError):
            continue
    return default


def _viewer_roster(ctx: Dict[str, Any], viewer_roster_id: Optional[str]) -> Optional[Dict[str, Any]]:
    rosters = ctx.get("rosters") or []
    if viewer_roster_id:
        for r in rosters:
            if str(r.get("roster_id")) == str(viewer_roster_id):
                return r
    return rosters[0] if rosters else None


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
    adp = _adp_map(is_sf)
    drafted = _drafted_round_map(platform, league_id)

    roster = _viewer_roster(ctx, viewer_roster_id) or {}
    player_ids = [str(p) for p in (roster.get("players") or [])]

    candidates: List[KeeperCandidate] = []
    for pid in player_ids:
        meta = players_index.get(pid) or {}
        name = meta.get("name") or f"Player {pid}"
        pos = (meta.get("pos") or meta.get("position") or "").upper()
        candidates.append(KeeperCandidate(
            player_id=pid,
            name=name,
            position=pos,
            drafted_round=drafted.get(pid),          # None → user sets it
            years_kept=0,                            # v1 default; editable in UI
            adp_overall=adp.get(pid),
            value=values.get(pid, 0.0),
        ))

    rules = KeeperRules(league_size=league_size, num_rounds=num_rounds)
    ranked = evaluate(candidates, rules, limit=max_keepers)

    seed = {
        "leagueSize": league_size,
        "numRounds": num_rounds,
        "maxKeepers": max_keepers,
        "isSuperflex": is_sf,
        "autoDraft": bool(drafted),   # did we auto-detect draft rounds?
        "platform": (platform or "sleeper").lower(),
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
