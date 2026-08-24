"""Tokenless *global* ADP fetchers for Yahoo, ESPN, and MyFantasyLeague.

These are the public, no-login ADP feeds. They are deliberately separate from the
OAuth/cookie league integrations (``yahoo_api``, ``espn_api``, ``mfl_api``): those
read a *specific user's league*, while this module reads each platform's *global*
draft market. Keeping them apart means a global-ADP change can never touch the
authenticated league paths, and vice-versa.

Everything here is normalized to ``canonical_id -> overall ADP`` (canonical =
Sleeper id, the repository's player key) via the existing crosswalk
infrastructure. Every network dependency is imported lazily so this module stays
importable in the lightweight CI suite, and every fetcher returns a structured
result and swallows failures into an empty payload so one provider's outage never
propagates.

None of these feeds is called on the request path — a central daily refresh
(``adp_service.refresh_global_adp_sources``) fetches them and persists snapshots,
and the resolver reads the snapshots. The audit script calls the fetchers
directly for verification.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_UA = {"User-Agent": "fantasy-dashboard/1.0 (+adp)"}
_DEFAULT_TIMEOUT = 20


# ── HTTP ──────────────────────────────────────────────────────────────────────
def _get_json(url: str, *, headers: Optional[dict] = None, params=None,
              timeout: int = _DEFAULT_TIMEOUT, retries: int = 2) -> Any:
    """GET ``url`` and parse JSON, with a small bounded retry.

    ``requests`` is imported lazily (the base CI job installs only pytest). Raises
    on exhausted retries so each fetcher can log and degrade to empty.
    """
    import requests  # lazy: keep module import light

    last_exc: Optional[Exception] = None
    hdrs = {**_UA, **(headers or {})}
    for attempt in range(max(1, retries)):
        try:
            resp = requests.get(url, headers=hdrs, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - normalized below
            last_exc = exc
            logger.debug("global_adp: GET failed (attempt %s) %s: %s",
                         attempt + 1, url, type(exc).__name__)
    assert last_exc is not None
    raise last_exc


def _empty(source: str, **meta) -> Dict[str, Any]:
    return {"source": source, "adp": {}, "extra": {}, "raw_count": 0,
            "mapped_count": 0, "unmapped": [], "meta": meta}


# ── Crosswalks (canonical = Sleeper id) ───────────────────────────────────────
_XWALK_CACHE: Dict[str, Dict[str, str]] = {}


def _sleeper_feed() -> Dict[str, Any]:
    from dashboard_services.api import get_nfl_players
    return get_nfl_players() or {}


def yahoo_id_to_canonical() -> Dict[str, str]:
    """yahoo_id -> canonical id, from Sleeper's player feed (carries yahoo_id)."""
    if "yahoo" in _XWALK_CACHE:
        return _XWALK_CACHE["yahoo"]
    out: Dict[str, str] = {}
    try:
        for sid, info in _sleeper_feed().items():
            yid = (info or {}).get("yahoo_id")
            if yid:
                out[str(yid)] = str(sid)
    except Exception:
        logger.debug("global_adp: yahoo crosswalk build failed", exc_info=True)
    if out:
        _XWALK_CACHE["yahoo"] = out
    return out


def espn_id_to_canonical() -> Dict[str, str]:
    """espn_id -> canonical id, merging Sleeper's feed with the players_index.

    Sleeper's ``espn_id`` is authoritative but *lags for recently-drafted players*
    — the ESPN ids of the newest stars (e.g. Gibbs/Chase/Nacua) are simply absent
    from the feed, so a Sleeper-only crosswalk drops exactly the top of the ADP
    board. The Tank01-derived players_index carries ``espnID`` for ~all players, so
    we merge it in to fill those gaps. Sleeper wins wherever both cover an id."""
    if "espn" in _XWALK_CACHE:
        return _XWALK_CACHE["espn"]
    out: Dict[str, str] = {}
    try:
        for sid, info in _sleeper_feed().items():
            eid = (info or {}).get("espn_id")
            if eid:
                out[str(eid)] = str(sid)
    except Exception:
        logger.debug("global_adp: espn crosswalk (sleeper) build failed", exc_info=True)
    # Merge (not just fallback-if-empty): add every players_index espnID that
    # Sleeper's feed didn't already map, which is where the recent players live.
    try:
        from utils.utils import load_players_index
        for cid, info in (load_players_index() or {}).items():
            eid = (info or {}).get("espnID") or (info or {}).get("espn_id")
            if eid and str(eid) not in out:
                out[str(eid)] = str(cid)
    except Exception:
        logger.debug("global_adp: espn crosswalk (index) merge failed", exc_info=True)
    if out:
        _XWALK_CACHE["espn"] = out
    return out


def _flip_comma_name(raw: str) -> str:
    """Reorder MFL's ``"Last, First"`` (optionally ``"Last, First Suffix"``) name
    to ``"First Last"`` so ``normalize_name`` lines it up with the ``"First Last"``
    players_index. ``normalize_name`` itself only strips periods/suffixes — it does
    NOT reorder the comma form — so without this every MFL name fails to match."""
    if "," not in (raw or ""):
        return raw or ""
    last, _, first = raw.partition(",")
    first, last = first.strip(), last.strip()
    return (f"{first} {last}".strip()) if first else last


def mfl_id_to_canonical(season: int) -> Dict[str, str]:
    """mfl_id -> canonical id, matched by normalized (name, position).

    MFL's global player export (no league id) lists every player with a stable
    MFL id; we match those to canonical ids by name/position, the same lossy
    fallback the league provider uses. Cached per process.
    """
    key = f"mfl:{int(season)}"
    if key in _XWALK_CACHE:
        return _XWALK_CACHE[key]
    out: Dict[str, str] = {}
    try:
        from utils.utils import load_players_index, normalize_name
        index = load_players_index() or {}
        by_name_pos: Dict[tuple, str] = {}
        by_name: Dict[str, str] = {}
        for cid, info in index.items():
            nm = normalize_name(info.get("full_name") or info.get("name") or "")
            pos = str(info.get("position") or info.get("pos") or "").upper()
            if nm:
                by_name_pos.setdefault((nm, pos), str(cid))
                by_name.setdefault(nm, str(cid))
        players = _mfl_player_rows(int(season))
        for p in players:
            raw_name = p.get("name") or ""
            # MFL names are "Last, First"; reorder before normalizing (normalize_name
            # does not handle the comma form) so they match the "First Last" index.
            nm = normalize_name(_flip_comma_name(raw_name))
            pos = str(p.get("position") or "").upper()
            cid = by_name_pos.get((nm, pos)) or by_name.get(nm)
            if cid and p.get("id"):
                out[str(p["id"])] = cid
    except Exception:
        logger.debug("global_adp: mfl crosswalk build failed", exc_info=True)
    if out:
        _XWALK_CACHE[key] = out
    return out


_NAME_INDEX_CACHE: Dict[str, tuple] = {}


def _name_pos_to_canonical() -> tuple:
    """(by_name_pos, by_name) name indexes over the players_index, for matching a
    feed that shares no id with our canonical space.

    Yahoo's public ADP carries a ``player_id`` that only maps through Sleeper's
    ``yahoo_id`` — which *lags for recent players*, dropping the whole top of the
    board — and the players_index has no yahoo id to merge in. Name/position is the
    only remaining bridge, so build it once and cache it. ``by_name_pos`` is tried
    first (position disambiguates same-named players); ``by_name`` is the fallback."""
    if "idx" in _NAME_INDEX_CACHE:
        return _NAME_INDEX_CACHE["idx"]
    by_name_pos: Dict[tuple, str] = {}
    by_name: Dict[str, str] = {}
    try:
        from utils.utils import load_players_index, normalize_name
        for cid, info in (load_players_index() or {}).items():
            nm = normalize_name((info or {}).get("name") or (info or {}).get("full_name") or "")
            pos = str((info or {}).get("pos") or (info or {}).get("position") or "").upper()
            if nm:
                by_name_pos.setdefault((nm, pos), str(cid))
                by_name.setdefault(nm, str(cid))
    except Exception:
        logger.debug("global_adp: name index build failed", exc_info=True)
    if by_name:
        _NAME_INDEX_CACHE["idx"] = (by_name_pos, by_name)
    return by_name_pos, by_name


def _match_name_pos(name, pos) -> Optional[str]:
    """canonical id for a ``name``/``pos``, or None. Position match preferred."""
    if not name:
        return None
    try:
        from utils.utils import normalize_name
        nm = normalize_name(name)
    except Exception:
        return None
    if not nm:
        return None
    by_name_pos, by_name = _name_pos_to_canonical()
    return by_name_pos.get((nm, str(pos or "").upper())) or by_name.get(nm)


def clear_crosswalk_cache() -> None:
    _XWALK_CACHE.clear()
    _NAME_INDEX_CACHE.clear()


def _mfl_player_rows(season: int) -> List[dict]:
    url = f"https://api.myfantasyleague.com/{int(season)}/export"
    data = _get_json(url, params={"TYPE": "players", "JSON": 1, "DETAILS": 1})
    players = ((data or {}).get("players") or {}).get("player") or []
    if isinstance(players, dict):
        players = [players]
    return [p for p in players if isinstance(p, dict)]


# ── Yahoo public global ADP ───────────────────────────────────────────────────
_YAHOO_URL = ("https://pub-api-ro.fantasysports.yahoo.com/fantasy/v2/game/nfl/"
              "players;sort=AR;start={start};count={count};out=draft_analysis"
              "?format=json_f")


def _iter_players_blocks(payload: Any):
    """Yield player entries from a Yahoo players collection.

    Yahoo nests the players under ``fantasy_content -> game -> players``. With
    ``format=json_f`` the current shape is a plain **list** of ``{"player": {...}}``
    dicts (``game`` itself a single dict, not a list). The older count-indexed
    ``{"count": N, "0": {"player": ...}, ...}`` dict shape is still tolerated.
    """
    fc = (payload or {}).get("fantasy_content") or {}
    game = fc.get("game")
    containers = game if isinstance(game, list) else [game]
    players_block = None
    for c in containers:
        if isinstance(c, dict) and "players" in c:
            players_block = c["players"]
            break
    if players_block is None and fc.get("players") is not None:
        players_block = fc["players"]
    # Current json_f shape: a list of {"player": {...}} entries.
    if isinstance(players_block, list):
        for entry in players_block:
            if isinstance(entry, dict):
                yield entry.get("player", entry)
        return
    # Legacy shape: a dict with a "count" and string-indexed entries.
    if not isinstance(players_block, dict):
        return
    count = players_block.get("count")
    try:
        count = int(count)
    except (TypeError, ValueError):
        count = None
    keys = [k for k in players_block.keys() if k != "count"]
    n = count if count is not None else len(keys)
    for i in range(n):
        entry = players_block.get(str(i))
        if isinstance(entry, dict):
            yield entry.get("player", entry)


def _flatten_yahoo(entry: Any) -> Dict[str, Any]:
    """Merge Yahoo's positional-list-of-single-key-dicts into one flat dict."""
    if isinstance(entry, dict):
        return entry
    flat: Dict[str, Any] = {}
    parts = entry if isinstance(entry, list) else [entry]
    for part in parts:
        if isinstance(part, dict):
            flat.update(part)
        elif isinstance(part, list):
            for sub in part:
                if isinstance(sub, dict):
                    flat.update(sub)
    return flat


def _yahoo_avg_pick(flat: Dict[str, Any]) -> Optional[float]:
    da = flat.get("draft_analysis")
    if isinstance(da, list):
        merged: Dict[str, Any] = {}
        for d in da:
            if isinstance(d, dict):
                merged.update(d)
        da = merged
    if not isinstance(da, dict):
        return None
    val = da.get("average_pick")
    try:
        ap = float(val)
        return ap if ap > 0 else None
    except (TypeError, ValueError):
        return None


def fetch_yahoo_global_adp(season: int, max_players: int = 350,
                           page: int = 25) -> Dict[str, Any]:
    """Public Yahoo global redraft ADP -> canonical id -> average pick.

    No OAuth. Scoring and QB format are mixed/global, so callers must record it as
    such. Empty payload on any failure.
    """
    result = _empty("yahoo", draft_type="redraft", scoring="mixed",
                    qb_format="mixed", scope="global")
    xwalk = yahoo_id_to_canonical()
    adp: Dict[str, float] = {}
    unmapped: List[str] = []
    raw = 0
    try:
        start = 0
        while start < max_players:
            payload = _get_json(_YAHOO_URL.format(start=start, count=page))
            got = 0
            for entry in _iter_players_blocks(payload):
                got += 1
                raw += 1
                flat = _flatten_yahoo(entry)
                ap = _yahoo_avg_pick(flat)
                if ap is None:
                    continue
                yid = flat.get("player_id")
                name = flat.get("name")
                if isinstance(name, dict):
                    name = name.get("full")
                cid = xwalk.get(str(yid)) if yid else None
                if not cid:
                    # Sleeper's yahoo_id lags for recent players (their ids are
                    # absent), so the id crosswalk drops the top of the board. Yahoo
                    # shares no other id with our canonical space, so fall back to a
                    # name/position match against the players_index.
                    cid = _match_name_pos(name, flat.get("display_position"))
                if cid:
                    adp[str(cid)] = ap
                elif len(unmapped) < 25:
                    unmapped.append(str(name or yid))
            if got < page:
                break
            start += page
    except Exception:
        logger.warning("global_adp: Yahoo global ADP fetch failed", exc_info=True)
        return result if not adp else {**result, "adp": adp}
    result.update(adp=adp, extra={}, raw_count=raw, mapped_count=len(adp),
                  unmapped=unmapped)
    return result


# ── ESPN public global ADP + PPR draft-room rank ──────────────────────────────
_ESPN_URL = ("https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/"
             "seasons/{season}/segments/0/leaguedefaults/3?view=kona_player_info")

# Offensive positions + K + D/ST slot ids for the X-Fantasy-Filter.
_ESPN_SLOT_IDS = [0, 2, 4, 6, 17, 16]
# defaultPositionId -> position, for the audit report only.
_ESPN_POS = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "D/ST"}


def _espn_filter(limit: int = 400) -> str:
    return json.dumps({"players": {
        "limit": int(limit),
        "sortDraftRanks": {"sortPriority": 100, "sortAsc": True, "value": "STANDARD"},
        "filterSlotIds": {"value": _ESPN_SLOT_IDS},
    }})


def fetch_espn_global_adp(season: int, limit: int = 400) -> Dict[str, Any]:
    """Public ESPN global ADP + a *separate* PPR draft-room rank.

    ``averageDraftPosition`` is ESPN's global ADP (shared across scoring defaults,
    so recorded as mixed/global — never labelled full-PPR). ``draftRanksByRankType
    .PPR.rank`` is ESPN's PPR platform rank; it is returned separately and must
    never be blended into ADP consensus. Empty payload on any failure.
    """
    result = _empty("espn", draft_type="redraft", scoring="mixed",
                    qb_format="mixed", scope="global")
    result["ppr_rank"] = {}
    xwalk = espn_id_to_canonical()
    adp: Dict[str, float] = {}
    ppr_rank: Dict[str, float] = {}
    unmapped: List[str] = []
    raw = 0
    try:
        payload = _get_json(_ESPN_URL.format(season=int(season)),
                            headers={"X-Fantasy-Filter": _espn_filter(limit)})
        players = (payload or {}).get("players") or []
        if isinstance(players, dict):
            players = list(players.values())
        for entry in players:
            if not isinstance(entry, dict):
                continue
            p = entry.get("player") if isinstance(entry.get("player"), dict) else entry
            raw += 1
            eid = p.get("id")
            cid = xwalk.get(str(eid)) if eid is not None else None
            own = p.get("ownership") or {}
            try:
                ap = float(own.get("averageDraftPosition"))
            except (TypeError, ValueError):
                ap = None
            ranks = p.get("draftRanksByRankType") or {}
            ppr_block = ranks.get("PPR") or {}
            try:
                pr = float(ppr_block.get("rank"))
            except (TypeError, ValueError):
                pr = None
            if not cid:
                if len(unmapped) < 25:
                    unmapped.append(str(p.get("fullName") or eid))
                continue
            if ap is not None and ap > 0:
                adp[str(cid)] = ap
            if pr is not None and pr > 0:
                ppr_rank[str(cid)] = pr
    except Exception:
        logger.warning("global_adp: ESPN global ADP fetch failed", exc_info=True)
        if adp or ppr_rank:
            return {**result, "adp": adp, "ppr_rank": ppr_rank}
        return result
    result.update(adp=adp, ppr_rank=ppr_rank, raw_count=raw,
                  mapped_count=len(adp), unmapped=unmapped)
    return result


# ── MyFantasyLeague free ADP export ───────────────────────────────────────────
_MFL_URL = "https://api.myfantasyleague.com/{season}/export"


def fetch_mfl_adp(season: int, *, is_ppr: Optional[int] = 1,
                  fcount: Optional[int] = 12, is_mock: Optional[int] = 0,
                  period: Optional[str] = None,
                  extra_params: Optional[dict] = None) -> Dict[str, Any]:
    """Free MFL ADP export -> canonical id -> average pick (+ min/max/pct).

    Only verified filters are sent: ``IS_PPR`` (scoring), ``FCOUNT`` (league
    size), ``IS_MOCK`` (real vs mock), ``PERIOD``. Dynasty/rookie/SF/TEP are NOT
    exposed by MFL's ADP filters and are recorded as unknown by callers. Empty
    payload on any failure.
    """
    meta = {"draft_type": "redraft", "scope": "global",
            "ppr": (1.0 if is_ppr else 0.0) if is_ppr is not None else "unknown",
            "qb_format": "1qb", "num_teams": fcount,
            "is_mock": is_mock}
    result = _empty("mfl", **meta)
    xwalk = mfl_id_to_canonical(int(season))
    params: Dict[str, Any] = {"TYPE": "adp", "JSON": 1}
    if is_ppr is not None:
        params["IS_PPR"] = int(is_ppr)
    if fcount is not None:
        params["FCOUNT"] = int(fcount)
    if is_mock is not None:
        params["IS_MOCK"] = int(is_mock)
    if period is not None:
        params["PERIOD"] = str(period)
    if extra_params:
        params.update(extra_params)
    adp: Dict[str, float] = {}
    extra: Dict[str, dict] = {}
    unmapped: List[str] = []
    raw = 0
    try:
        payload = _get_json(_MFL_URL.format(season=int(season)), params=params)
        rows = ((payload or {}).get("adp") or {}).get("player") or []
        if isinstance(rows, dict):
            rows = [rows]
        for r in rows:
            if not isinstance(r, dict):
                continue
            raw += 1
            mid = r.get("id")
            cid = xwalk.get(str(mid)) if mid is not None else None
            try:
                ap = float(r.get("averagePick"))
            except (TypeError, ValueError):
                ap = None
            if not cid:
                if len(unmapped) < 25:
                    unmapped.append(str(mid))
                continue
            if ap is None or ap <= 0:
                continue
            adp[str(cid)] = ap
            extra[str(cid)] = {
                "min_pick": _f(r.get("minPick")),
                "max_pick": _f(r.get("maxPick")),
                "draft_pct": _f(r.get("draftSelPct")),
            }
    except Exception:
        logger.warning("global_adp: MFL ADP fetch failed", exc_info=True)
        return result if not adp else {**result, "adp": adp, "extra": extra}
    result.update(adp=adp, extra=extra, raw_count=raw, mapped_count=len(adp),
                  unmapped=unmapped)
    return result


def _f(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
