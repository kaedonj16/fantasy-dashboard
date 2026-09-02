"""Keeper Assistant page.

Server-renders the "who should I keep?" tool for a league: an optimizer that
picks the best keepers under the league limit, with a full sortable table
available as a toggle. All the decision math lives in utils.keeper_value (pure +
unit-tested); this module just assembles real roster / draft / ADP / value data
into candidates and hands them to the client, which re-runs the same math live
as the manager tweaks the keeper limit and cost rules.

Draft-round auto-detection uses Sleeper's season-chain draft results, Yahoo's
draftresults feed, and ESPN's completed draft (League.draft / mDraftDetail).
Waiver adds anywhere still start blank so the manager can set the cost.
Auction/FAAB dollars are imported when providers expose them (MFL pick amounts);
otherwise the $ field stays editable.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from utils.keeper_value import (
    KeeperRules, KeeperCandidate, evaluate, project_league_keepers,
)

logger = logging.getLogger(__name__)

_VERDICT_LABEL = {"keep": "KEEP", "toss": "TOSS-UP", "pass": "PASS"}


def years_kept_from_draft_season(draft_season, current_season) -> int:
    """Seasons already kept: current - draft_season - 1, floored at 0.

    A player drafted last year is about to be kept for the first time (0).
    ESPN only exposes a keeper boolean for the current draft, so that path
    uses 1 vs 0 rather than a multi-year count.
    """
    try:
        ds = int(draft_season)
        cs = int(current_season)
    except (TypeError, ValueError):
        return 0
    if ds <= 0 or cs <= 0:
        return 0
    return max(0, cs - ds - 1)


def _is_superflex(ctx: Dict[str, Any]) -> bool:
    from utils.lineup_slots import is_superflex_lineup
    positions = ctx.get("roster_positions") or ctx.get("roster_positions_list") or []
    return is_superflex_lineup(positions)


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


def _parse_espn_draft_picks(picks, espn_to_canon: Optional[Dict[str, str]] = None) -> Dict[str, int]:
    """player_id -> drafted round from espn_api Pick objects or mDraftDetail dicts."""
    return _parse_espn_draft_meta(picks, espn_to_canon)[0]


def _parse_espn_draft_meta(picks, espn_to_canon: Optional[Dict[str, str]] = None):
    """(player_id -> round, player_id -> years_kept) from ESPN draft picks."""
    canon = espn_to_canon or {}
    drafted: Dict[str, int] = {}
    years_kept: Dict[str, int] = {}
    for p in picks or []:
        if isinstance(p, dict):
            espn_pid = p.get("playerId") or p.get("player_id")
            rnd = p.get("roundId") or p.get("round_num") or p.get("round")
            keeper = p.get("keeper") or p.get("isKeeper")
        else:
            espn_pid = getattr(p, "playerId", None) or getattr(p, "player_id", None)
            rnd = (
                getattr(p, "round_num", None)
                or getattr(p, "roundId", None)
                or getattr(p, "round", None)
            )
            keeper = getattr(p, "keeper", None) or getattr(p, "isKeeper", None)
        if not espn_pid or not rnd:
            continue
        try:
            rnd_i = int(rnd)
        except (TypeError, ValueError):
            continue
        if rnd_i <= 0:
            continue
        pid = canon.get(str(espn_pid)) or str(espn_pid)
        if pid not in drafted:
            drafted[pid] = rnd_i
            years_kept[pid] = 1 if keeper else 0
    return drafted, years_kept


def _espn_draft_maps(league_id: str, season: int):
    """(player_id -> round, player_id -> years_kept) from one ESPN draft fetch."""
    try:
        from dashboard_services.providers import espn_api
        picks = espn_api.iter_draft_picks(int(season), str(league_id))
        if not picks:
            return {}, {}
        try:
            canon = espn_api._espn_to_canon_cached()
        except Exception:
            canon = {}
        return _parse_espn_draft_meta(picks, canon)
    except Exception:
        logger.debug("[keeper] espn draft load failed", exc_info=True)
        return {}, {}


def _espn_drafted_round_map(league_id: str, season: int) -> Dict[str, int]:
    """player_id -> drafted round for an ESPN league."""
    return _espn_draft_maps(league_id, season)[0]


def _espn_years_kept_map(league_id: str, season: int) -> Dict[str, int]:
    """player_id -> 1 if ESPN marked the pick as a keeper, else 0."""
    return _espn_draft_maps(league_id, season)[1]


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
    """(player_id -> most recent drafted round, deepest completed draft rounds,
    player_id -> years_kept).

    Walks the league's season chain newest-first. In the offseason the current
    season's draft has not happened yet (pre_draft, no picks), so the rounds a
    roster was actually built in live under a previous season's league. Each
    player keeps the round from the most recent draft that took him, and the
    round scale comes from the deepest completed draft found (a startup/full
    draft rather than a small rookie draft). years_kept is current_season minus
    that draft's season minus 1 (first keep = 0)."""
    drafted: Dict[str, int] = {}
    years_kept: Dict[str, int] = {}
    deepest = 0
    try:
        from dashboard_services.api import get_drafts, get_draft_picks, build_league_history_map
    except Exception:
        logger.debug("[keeper] sleeper draft api unavailable", exc_info=True)
        return drafted, deepest, years_kept

    lid_to_year: Dict[str, int] = {}
    try:
        hist = build_league_history_map("sleeper", str(league_id), int(season or 0)) or {}
        for yr, lid in hist.items():
            try:
                lid_to_year[str(lid)] = int(yr)
            except (TypeError, ValueError):
                continue
    except Exception:
        logger.debug("[keeper] league history map failed", exc_info=True)

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
            draft_season = lid_to_year.get(str(lid), season)
            for p in picks:
                pid = str(p.get("player_id") or "")
                rnd = p.get("round")
                if not pid or not rnd or pid in drafted:
                    continue   # first hit wins: newest season, deepest draft
                try:
                    drafted[pid] = int(rnd)
                    years_kept[pid] = years_kept_from_draft_season(draft_season, season)
                except (TypeError, ValueError):
                    continue
    return drafted, deepest, years_kept


def _drafted_round_map(platform: str, league_id: str, season: int = 0) -> Dict[str, int]:
    """player_id -> the round they were drafted, for Sleeper, Yahoo, or ESPN.

    Sleeper walks the league's season chain for completed drafts; Yahoo reads the
    league draftresults resource; ESPN reads League.draft / mDraftDetail. Empty
    for other platforms (players show as undrafted and users set costs manually)."""
    plat = (platform or "").lower()
    if plat == "yahoo":
        return _yahoo_drafted_round_map(league_id, season)
    if plat == "espn":
        return _espn_drafted_round_map(league_id, season)
    if plat != "sleeper":
        return {}
    return _sleeper_draft_history(league_id, season)[0]


def _coerce_auction_amount(raw: Any) -> Optional[float]:
    """Parse a provider auction/FAAB dollar field into a positive float."""
    if raw in (None, "", 0, "0"):
        return None
    if isinstance(raw, str):
        raw = raw.strip().lstrip("$").replace(",", "")
        if not raw:
            return None
    try:
        amt = float(raw)
    except (TypeError, ValueError):
        return None
    return amt if amt > 0 else None


def parse_auction_amounts_from_picks(picks: Optional[List[Any]]) -> Dict[str, float]:
    """player_id -> auction $ from a flat pick list (pure).

    Reads MFL ``metadata.auction_amount``, Sleeper ``metadata.amount``, and
    common top-level amount / bidAmount fields when present.
    """
    out: Dict[str, float] = {}
    for p in picks or []:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("player_id") or p.get("playerId") or "").strip()
        if not pid:
            continue
        meta = p.get("metadata") if isinstance(p.get("metadata"), dict) else {}
        raw = None
        for src in (
            meta.get("auction_amount"),
            meta.get("amount"),
            p.get("auction_amount"),
            p.get("amount"),
            p.get("bidAmount"),
            p.get("bid_amount"),
        ):
            if src is not None and src != "":
                raw = src
                break
        amt = _coerce_auction_amount(raw)
        if amt is None:
            continue
        # Prefer the most recent / last-seen amount if a player appears twice.
        out[pid] = amt
    return out


def parse_auction_amounts_from_drafts(drafts: Optional[List[Dict[str, Any]]]) -> Dict[str, float]:
    """player_id -> auction $ from provider draft pick metadata (pure)."""
    out: Dict[str, float] = {}
    for d in drafts or []:
        out.update(parse_auction_amounts_from_picks(d.get("picks") or []))
    return out


def _hydrate_sleeper_draft_picks(drafts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach picks to Sleeper draft shells (list endpoint omits pick rows)."""
    try:
        from dashboard_services.api import get_draft_picks
    except Exception:
        return drafts
    hydrated: List[Dict[str, Any]] = []
    for d in drafts or []:
        row = dict(d)
        if row.get("picks"):
            hydrated.append(row)
            continue
        did = row.get("draft_id")
        if not did:
            hydrated.append(row)
            continue
        try:
            row["picks"] = get_draft_picks(str(did)) or []
        except Exception:
            logger.debug("[keeper] sleeper picks for auction costs failed", exc_info=True)
            row["picks"] = []
        hydrated.append(row)
    return hydrated


def _auction_cost_map(
    platform: str,
    league_id: str,
    season: int,
    drafts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, float]:
    """Best-effort auction $ paid map for a league.

    MFL embeds amounts on draft picks. Sleeper returns draft shells without
    picks — hydrate via get_draft_picks when needed. ESPN/Yahoo rarely expose
    bid amounts on the normalized draft list; the UI stays editable.
    """
    plat = (platform or "").lower()
    if not league_id and drafts is None:
        return {}
    if drafts is None:
        try:
            from dashboard_services.platform_api import get_drafts
            drafts = get_drafts(plat, str(league_id), int(season)) or []
        except Exception:
            logger.debug("[keeper] drafts for auction costs failed", exc_info=True)
            return {}
    costs = parse_auction_amounts_from_drafts(drafts)
    if costs:
        return costs
    if plat == "sleeper":
        return parse_auction_amounts_from_drafts(_hydrate_sleeper_draft_picks(list(drafts or [])))
    return {}


def _num_rounds(platform: str, league_id: str, default: int = 15,
                drafted: Optional[Dict[str, int]] = None, deepest: int = 0) -> int:
    """Draft rounds for the keeper-cost scale.

    Uses the startup/full draft's round count; defaults to a standard redraft
    depth when it can't be detected or looks like a small rookie-only draft
    (which would make the undrafted cost absurdly cheap). ``deepest`` is the
    round count found while loading Sleeper picks; Yahoo and ESPN have no
    reliable round count in their draft list, so they derive the scale from
    the deepest drafted round."""
    plat = (platform or "").lower()
    if plat in ("yahoo", "espn"):
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
    """(drafted_round_map, num_rounds, years_kept_map) for a league, in one pass.

    Keeps the Sleeper season-chain walk to a single fetch instead of doing it
    once for the picks and again for the round count. Callers that only unpack
    two values still work against older test stubs.
    """
    plat = (platform or "").lower()
    if plat == "sleeper":
        drafted, deepest, years = _sleeper_draft_history(league_id, season)
        return drafted, _num_rounds(plat, league_id, drafted=drafted, deepest=deepest), years
    if plat == "espn":
        drafted, years = _espn_draft_maps(league_id, season)
        return drafted, _num_rounds(plat, league_id, drafted=drafted), years
    drafted = _drafted_round_map(plat, league_id, season)
    return drafted, _num_rounds(plat, league_id, drafted=drafted), {}


def _unpack_draft_context(result) -> tuple:
    """Accept 2-tuple test stubs or the 3-tuple (drafted, rounds, years_kept)."""
    drafted = result[0] if result else {}
    num_rounds = result[1] if result and len(result) > 1 else 15
    years = result[2] if result and len(result) > 2 else {}
    return drafted, num_rounds, years or {}


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


def _league_type_code(ctx: Dict[str, Any]) -> Optional[int]:
    """Sleeper's league type: 0 redraft, 1 keeper, 2 dynasty. None if unknown.

    Only Sleeper publishes this; ESPN and Yahoo have no dynasty flag, so those
    leagues stay unknown and are treated as keeper-capable."""
    for src in (ctx.get("league_settings"), ctx.get("settings")):
        try:
            v = (src or {}).get("type")
            if v is not None:
                return int(v)
        except (TypeError, ValueError, AttributeError):
            continue
    return None


def is_dynasty_without_keepers(ctx: Dict[str, Any]) -> bool:
    """True for a true dynasty league: you keep your whole roster, so there is
    no keeper decision to make.

    The surplus model prices a keeper by the round he was drafted in. Dynasty
    rosters are built from a startup years back plus rookie drafts, trades and
    waivers, so most players have no drafted round in the current league and all
    of them collapse onto the undrafted default - the last round. That produced
    a page of identical, meaningless costs (and inflated surpluses to match).
    Rather than show numbers that look real, the page explains itself.

    A dynasty league that *does* configure a keeper limit is a real keeper
    league and keeps the tool."""
    return _league_type_code(ctx) == 2 and _detected_keeper_limit(ctx) == 0


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
    years_kept: Optional[Dict[str, int]] = None,
) -> List[KeeperCandidate]:
    """Build keeper candidates for a set of player ids (one team's roster).
    Market ADP falls back to the redraft value rank when it's missing."""
    value_rank = value_rank or {}
    years_kept = years_kept or {}
    out: List[KeeperCandidate] = []
    for pid in player_ids:
        meta = players_index.get(pid) or {}
        try:
            yk = int(years_kept.get(pid) or 0)
        except (TypeError, ValueError):
            yk = 0
        out.append(KeeperCandidate(
            player_id=pid,
            name=meta.get("name") or f"Player {pid}",
            position=(meta.get("pos") or meta.get("position") or "").upper(),
            drafted_round=drafted.get(pid),   # None → UI/user sets it
            years_kept=max(0, yk),
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
    drafted, num_rounds, years_kept = _unpack_draft_context(
        _draft_context(platform, league_id, _season)
    )
    _ro = rules_override or {}

    def _rule_int(key, lo, hi, default=None):
        try:
            v = int(_ro[key])
        except (KeyError, TypeError, ValueError):
            return default
        return max(lo, min(hi, v))

    # one_per_round defaults on (matches the keeper page); the handoff can turn it
    # off for leagues that let two keepers share a round.
    _opr = _ro.get("one_per_round", True) if _ro else True
    rules = KeeperRules(
        league_size=league_size,
        num_rounds=num_rounds,
        round_offset=_rule_int("round_offset", -5, 5, 0) or 0,
        escalation=_rule_int("escalation", 0, 5, 1) if _ro else 1,
        undrafted_round=_rule_int("undrafted_round", 1, num_rounds),
        one_per_round=bool(_opr),
    )

    per_team: Dict[str, List[KeeperCandidate]] = {}
    for r in (ctx.get("rosters") or []):
        rid = str(r.get("roster_id"))
        pids = [str(p) for p in (r.get("players") or [])]
        per_team[rid] = _candidates_for_ids(
            pids, players_index, values, adp, drafted, value_rank, years_kept,
        )

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
            # cost_round is already final here: project_league_keepers ran the
            # optimizer (which analyzes every candidate and, under one_per_round,
            # resolves cost collisions) over these same objects in place. Re-running
            # analyze would recompute the raw cost and undo any collision bump.
            # "projected" means a rival team's estimated keepers. The viewer's
            # own roster is always "yours" — whether those ids came from the
            # optimizer or an explicit viewer_kept_ids override — so the draft
            # room banner can count ownership without a separate handoff.
            kept.append({
                "id": pid, "name": c.name, "pos": c.position,
                "rosterId": rid, "costRound": c.cost_round,
                "projected": not (vr is not None and rid == vr),
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
    force: bool = False,
) -> str:
    """Return the Keeper Assistant page body HTML for a league context.

    ``adp_source`` picks which market ADP feeds the surplus model (keeper
    decisions are redraft, so the redraft sources apply: consensus / sleeper /
    yahoo). The page's source dropdown reloads with ?adp_source=<v>.

    ``season`` is the route's season. It is passed explicitly because the
    Draft Room handoff link needs one: when a cached ctx comes back without a
    season the link used to render empty, which silently dropped the whole
    "Open in Draft Room" button and left no way to carry keepers over.

    A true dynasty league keeps every player, so the tool explains itself
    instead of rendering placeholder costs; ``force`` (from ?show=1) overrides
    that for a dynasty league that runs informal keepers."""
    from dashboard_services.pages._keeper_render import (  # local import: keeps this module import-light
        render_keeper_html, render_dynasty_notice_html,
    )

    _plat_early = (platform or "sleeper").lower()
    _season_early = int(ctx.get("season") or season or 0)
    if not force and is_dynasty_without_keepers(ctx):
        _durl = (f"/{_plat_early}/{_season_early}/{league_id}/draft"
                 if (league_id and _season_early) else "")
        return render_dynasty_notice_html(draft_url=_durl, show_anyway_url="?show=1")

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
    drafted, num_rounds, years_kept = _unpack_draft_context(
        _draft_context(platform, league_id, _season)
    )

    roster = _viewer_roster(ctx, viewer_roster_id) or {}
    player_ids = [str(p) for p in (roster.get("players") or [])]

    candidates = _candidates_for_ids(
        player_ids, players_index, values, adp, drafted, value_rank, years_kept,
    )
    rules = KeeperRules(league_size=league_size, num_rounds=num_rounds, one_per_round=True)
    ranked = evaluate(candidates, rules, limit=max_keepers)

    _plat = (platform or "sleeper").lower()
    # Reuses the resolved season above (ctx, else the route's). Deriving it from
    # ctx alone here silently produced an empty link - and therefore no
    # "Open in Draft Room" button - whenever the cached ctx had no season.
    draft_url = (f"/{_plat}/{_season}/{league_id}/draft?keepers=1"
                 if (league_id and _season) else "")
    try:
        from dashboard_services.adp_service import adp_source_options
        _src_opts = [{"value": v, "label": l} for v, l in adp_source_options("redraft", _season)]
    except Exception:
        _src_opts = []
    seed = {
        "leagueSize": league_size,
        "numRounds": num_rounds,
        "maxKeepers": max_keepers,
        "isSuperflex": is_sf,
        "onePerRound": True,          # default: no two keepers share a cost round
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
    try:
        from utils.league_format import detect_league_format
        from dashboard_services.platform_api import get_drafts
        _drafts = get_drafts(_plat, league_id, _season) or []
        _fmt = detect_league_format(
            league=ctx.get("league") or {},
            drafts=_drafts,
            settings=ctx.get("league_settings") or (ctx.get("league") or {}).get("settings"),
        )
        seed["isAuction"] = bool(_fmt.get("is_auction"))
        seed["auctionBudget"] = _fmt.get("auction_budget")
        costs = (
            _auction_cost_map(_plat, league_id, _season, drafts=_drafts)
            if seed["isAuction"] else {}
        )
        imported = 0
        for pl in seed["players"]:
            amt = costs.get(str(pl["id"]))
            if amt is not None:
                pl["auctionCost"] = round(float(amt), 2)
                imported += 1
            else:
                pl["auctionCost"] = None
        seed["auctionCostsImported"] = imported > 0
    except Exception:
        seed["isAuction"] = False
        seed["auctionBudget"] = None
        seed["auctionCostsImported"] = False
    return render_keeper_html(seed)
