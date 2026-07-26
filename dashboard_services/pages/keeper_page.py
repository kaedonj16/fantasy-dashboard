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


def _yahoo_drafted_round_map(league_id: str, season: int) -> Dict[str, int]:
    """player_id -> drafted round for a Yahoo league (via draftresults)."""
    try:
        from dashboard_services.providers.yahoo_api import get_league_token, get_draft_results
        token = get_league_token(str(league_id), int(season))
        if not token:
            return {}
        return get_draft_results(int(season), str(league_id), token) or {}
    except Exception:
        logger.debug("[keeper] yahoo draft load failed", exc_info=True)
        return {}


def _sleeper_league_chain(league_id: str, season: int, max_seasons: int = 5) -> List[str]:
    """This league's id plus its prior seasons' ids, newest season first.

    Sleeper gives every season its own league_id (linked by previous_league_id),
    so the current league only knows about the current season's draft."""
    ids: List[str] = [str(league_id)]
    try:
        from dashboard_services.api import build_league_history_map
        hist = build_league_history_map("sleeper", str(league_id), int(season or 0)) or {}
        for _yr in sorted(hist.keys(), reverse=True):
            lid = str(hist[_yr])
            if lid not in ids:
                ids.append(lid)
    except Exception:
        logger.debug("[keeper] league history walk failed", exc_info=True)
    return ids[:max_seasons]


def _sleeper_draft_history(league_id: str, season: int) -> tuple:
    """(player_id -> most recent drafted round, deepest completed draft rounds).

    Walks the league's season chain newest-first. In the offseason the current
    season's draft has not happened yet (pre_draft, no picks), so the rounds a
    roster was actually built in live under a previous season's league. Each
    player keeps the round from the most recent draft that took him, and the
    round scale comes from the deepest completed draft found (a startup/full
    draft rather than a small rookie draft)."""
    drafted: Dict[str, int] = {}
    deepest = 0
    try:
        from dashboard_services.api import get_drafts, get_draft_picks
    except Exception:
        logger.debug("[keeper] sleeper draft api unavailable", exc_info=True)
        return drafted, deepest

    for lid in _sleeper_league_chain(league_id, season):
        try:
            drafts = get_drafts(lid) or []
        except Exception:
            logger.debug("[keeper] draft list failed for %s", lid, exc_info=True)
            continue
        # Completed drafts only (a scheduled draft has no picks), deepest first
        # so the full draft sets the round scale before any rookie draft.
        done = [d for d in drafts if str(d.get("status")) == "complete"]
        for d in sorted(done, key=_draft_rounds, reverse=True):
            try:
                picks = get_draft_picks(str(d.get("draft_id"))) or []
            except Exception:
                logger.debug("[keeper] picks failed for %s", d.get("draft_id"), exc_info=True)
                continue
            if not picks:
                continue
            deepest = max(deepest, _draft_rounds(d))
            for p in picks:
                pid = str(p.get("player_id") or "")
                rnd = p.get("round")
                if not pid or not rnd or pid in drafted:
                    continue   # first hit wins: newest season, deepest draft
                try:
                    drafted[pid] = int(rnd)
                except (TypeError, ValueError):
                    continue
    return drafted, deepest


def _drafted_round_map(platform: str, league_id: str, season: int = 0) -> Dict[str, int]:
    """player_id -> the round they were drafted, for Sleeper or Yahoo leagues.

    Sleeper walks the league's season chain for completed drafts; Yahoo reads the
    league draftresults resource. Empty for other platforms (players show as
    undrafted and users set costs manually)."""
    plat = (platform or "").lower()
    if plat == "yahoo":
        return _yahoo_drafted_round_map(league_id, season)
    if plat != "sleeper":
        return {}
    return _sleeper_draft_history(league_id, season)[0]


def _num_rounds(platform: str, league_id: str, default: int = 15,
                drafted: Optional[Dict[str, int]] = None, deepest: int = 0) -> int:
    """Draft rounds for the keeper-cost scale.

    Uses the startup/full draft's round count; defaults to a standard redraft
    depth when it can't be detected or looks like a small rookie-only draft
    (which would make the undrafted cost absurdly cheap). ``deepest`` is the
    round count found while loading Sleeper picks; Yahoo has no round count in
    its draft list, so it derives the scale from the deepest drafted round."""
    plat = (platform or "").lower()
    if plat == "yahoo":
        rounds = max(drafted.values()) if drafted else 0
        return rounds if rounds >= 8 else default
    if plat != "sleeper":
        return default
    if deepest >= 8:
        return deepest
    # No usable round count from the picks walk: fall back to the league's own
    # draft list, then to standard depth.
    try:
        from dashboard_services.api import get_drafts
        rounds = _draft_rounds(_best_draft(get_drafts(league_id) or []))
        return rounds if rounds >= 8 else default
    except Exception:
        return default


def _draft_context(platform: str, league_id: str, season: int) -> tuple:
    """(drafted_round_map, num_rounds) for a league, in one pass.

    Keeps the Sleeper season-chain walk to a single fetch instead of doing it
    once for the picks and again for the round count."""
    plat = (platform or "").lower()
    if plat == "sleeper":
        drafted, deepest = _sleeper_draft_history(league_id, season)
        return drafted, _num_rounds(plat, league_id, drafted=drafted, deepest=deepest)
    drafted = _drafted_round_map(plat, league_id, season)
    return drafted, _num_rounds(plat, league_id, drafted=drafted)


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
    limit_override: Optional[int] = None,
    rules_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """League-wide keeper set for the draft board.

    Every team's likely keepers are projected with the surplus optimizer; the
    viewer's team is replaced by their *actual* selections when ``viewer_kept_ids``
    is supplied (the handoff from the keeper page). Returns a payload the draft
    room seeds into __draftCfg.keepers:

        {limit, viewerRoster, autoDraft,
         byTeam: {roster_id: [player_id, ...]},
         kept:   [{id, name, pos, rosterId, costRound, projected}, ...]}

    ``limit_override`` is the keeper limit the user is actually playing by (the
    keeper page's "Keep up to N", carried over in the handoff). Without it every
    other team was projected against the league default, so a user keeping 3
    still saw rivals holding far fewer.

    ``rules_override`` carries the same page's cost rules (undrafted round,
    round offset, escalation). The undrafted round matters most: left at the
    default it is the last round, so every player with no drafted round - which
    on a dynasty roster is most of them - prices identically at the deepest
    round the league has ever drafted.
    """
    is_sf = _is_superflex(ctx)
    try:
        league_size = int(ctx.get("total_rosters") or len(ctx.get("rosters") or []) or 12)
    except (TypeError, ValueError):
        league_size = 12
    _season = int(ctx.get("season") or 0)
    try:
        limit = int(limit_override) if limit_override else _max_keepers(ctx)
    except (TypeError, ValueError):
        limit = _max_keepers(ctx)
    limit = max(0, min(limit, 25))   # sane bound; the value arrives from a query param

    players_index: Dict[str, Any] = {}
    try:
        from utils.utils import load_players_index
        players_index = load_players_index() or {}
    except Exception:
        logger.debug("[keeper] players_index load failed", exc_info=True)
    values = _redraft_value_map(is_sf)
    adp = _adp_map(is_sf, _season)
    value_rank = _value_rank_map(values)
    drafted, num_rounds = _draft_context(platform, league_id, _season)
    _ro = rules_override or {}

    def _rule_int(key, lo, hi, default=None):
        try:
            v = int(_ro[key])
        except (KeyError, TypeError, ValueError):
            return default
        return max(lo, min(hi, v))

    rules = KeeperRules(
        league_size=league_size,
        num_rounds=num_rounds,
        round_offset=_rule_int("round_offset", -5, 5, 0) or 0,
        escalation=_rule_int("escalation", 0, 5, 1) if _ro else 1,
        undrafted_round=_rule_int("undrafted_round", 1, num_rounds),
    )

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
    adp_source: str = "consensus",
    season: Optional[int] = None,
) -> str:
    """Return the Keeper Assistant page body HTML for a league context.

    ``adp_source`` picks which market ADP feeds the surplus model (keeper
    decisions are redraft, so the redraft sources apply: consensus / sleeper /
    yahoo). The page's source dropdown reloads with ?adp_source=<v>.

    ``season`` is the route's season. It is passed explicitly because the
    Draft Room handoff link needs one: when a cached ctx comes back without a
    season the link used to render empty, which silently dropped the whole
    "Open in Draft Room" button and left no way to carry keepers over."""
    from dashboard_services.pages._keeper_render import render_keeper_html  # local import: keeps this module import-light

    is_sf = _is_superflex(ctx)
    try:
        league_size = int(ctx.get("total_rosters") or len(ctx.get("rosters") or []) or 12)
    except (TypeError, ValueError):
        league_size = 12
    # Prefer the ctx season, but fall back to the route's so a ctx without one
    # still yields a working Draft Room link (and a real ADP season).
    _season = int(ctx.get("season") or season or 0)
    max_keepers = _max_keepers(ctx)

    players_index = {}
    try:
        from utils.utils import load_players_index
        players_index = load_players_index() or {}
    except Exception:
        logger.debug("[keeper] players_index load failed", exc_info=True)

    values = _redraft_value_map(is_sf)
    adp = _adp_map(is_sf, _season, source=adp_source)
    value_rank = _value_rank_map(values)
    drafted, num_rounds = _draft_context(platform, league_id, _season)

    roster = _viewer_roster(ctx, viewer_roster_id) or {}
    player_ids = [str(p) for p in (roster.get("players") or [])]

    candidates = _candidates_for_ids(player_ids, players_index, values, adp, drafted, value_rank)
    rules = KeeperRules(league_size=league_size, num_rounds=num_rounds)
    ranked = evaluate(candidates, rules, limit=max_keepers)

    _plat = (platform or "sleeper").lower()
    # Reuses the resolved season above (ctx, else the route's). Deriving it from
    # ctx alone here silently produced an empty link - and therefore no
    # "Open in Draft Room" button - whenever the cached ctx had no season.
    draft_url = (f"/{_plat}/{_season}/{league_id}/draft?keepers=1"
                 if (league_id and _season) else "")
    try:
        from dashboard_services.adp_service import adp_source_options
        _src_opts = [{"value": v, "label": l} for v, l in adp_source_options("redraft")]
    except Exception:
        _src_opts = []
    seed = {
        "leagueSize": league_size,
        "numRounds": num_rounds,
        "maxKeepers": max_keepers,
        "isSuperflex": is_sf,
        "autoDraft": bool(drafted),   # did we auto-detect draft rounds?
        "platform": _plat,
        "leagueId": str(league_id or ""),
        "draftUrl": draft_url,
        "adpSource": adp_source,
        "adpSourceOptions": _src_opts,
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
