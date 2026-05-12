# dashboard_services/espn_api.py
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from espn_api.football import League

from utils.utils import load_players_index


# ============================================================
# Errors
# ============================================================

class ESPNError(Exception):
    pass


# ============================================================
# Env + helpers
# ============================================================

def _require_env(name: str) -> str:
    val = (os.getenv(name, "") or "").strip()
    if not val:
        raise ESPNError(f"Missing required env var: {name}")
    return val


def _normalize_swid(swid: str) -> str:
    swid = (swid or "").strip()
    if swid and not (swid.startswith("{") and swid.endswith("}")):
        swid = "{" + swid.strip("{}") + "}"
    return swid


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return default if x is None else float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return default if x is None else int(x)
    except Exception:
        return default


def _split_points(val: Any) -> tuple[int, int]:
    f = safe_float(val, 0.0)
    whole = int(f)
    dec = int(round((f - whole) * 100))
    return whole, dec


def _streak_from_outcomes(outcomes: Any) -> str:
    if not isinstance(outcomes, list):
        return ""

    cleaned = [x for x in (str(o).upper() for o in outcomes) if x in ("W", "L", "T")]
    if not cleaned:
        return ""

    last = cleaned[-1]
    n = 0
    for x in reversed(cleaned):
        if x != last:
            break
        n += 1
    return f"{n}{last}"


# ============================================================
# GLOBAL CACHES (CRITICAL)
# ============================================================

@lru_cache(maxsize=16)
def _league_cached(season: int, league_id: str) -> League:
    # Credentials are optional - public leagues work without them;
    # private leagues require ESPN_S2 + ESPN_SWID in the environment.
    espn_s2 = (os.getenv("ESPN_S2") or "").strip() or None
    swid = _normalize_swid(os.getenv("ESPN_SWID", "")) or None
    return League(
        league_id=int(league_id),
        year=int(season),
        espn_s2=espn_s2,
        swid=swid,
    )


def _league(season: int, league_id: str) -> League:
    return _league_cached(season, league_id)


@lru_cache(maxsize=1)
def _players_index_cached() -> Dict[str, Dict[str, Any]]:
    return load_players_index()


@lru_cache(maxsize=1)
def _espn_to_canon_cached() -> Dict[str, str]:
    return build_espn_to_canonical(_players_index_cached())


@lru_cache(maxsize=128)
def _box_scores_cached(season: int, league_id: str, week: int):
    return _league(season, league_id).box_scores(week)


@lru_cache(maxsize=16)
def _playoff_schedule_cached(season: int, league_id: str) -> List[Dict[str, Any]]:
    lg = _league(season, league_id)
    data = lg.espn_request.league_get(params={"view": "mMatchup"})
    return data.get("schedule") or []


# ============================================================
# Public API
# ============================================================

def get_league(season: int, league_id: str) -> Dict[str, Any]:
    lg = _league(season, league_id)
    name = (
            getattr(getattr(lg, "settings", None), "name", None)
            or getattr(lg, "name", None)
            or "ESPN League"
    )
    return {
        "league_id": str(league_id),
        "season": int(season),
        "name": name,
    }


def get_users(season: int, league_id: str) -> List[Dict[str, Any]]:
    lg = _league(season, league_id)
    swid = _normalize_swid(os.getenv("ESPN_SWID", ""))

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for t in lg.teams or []:
        team_name = getattr(t, "team_name", None) or getattr(t, "name", None)
        logo = getattr(t, "logo_url", None) or getattr(t, "logo", None)
        owners = getattr(t, "owners", None) or []

        for o in owners:
            oid = str(o.get("id") or "").strip()
            if not oid or oid in seen:
                continue
            seen.add(oid)

            display = (
                    o.get("displayName")
                    or o.get("firstName")
                    or f"User {oid}"
            )

            out.append({
                "avatar": logo,
                "display_name": display,
                "is_bot": False,
                "is_owner": (swid == oid) if swid else None,
                "league_id": str(league_id),
                "metadata": {"team_name": team_name} if team_name else {},
                "settings": None,
                "user_id": oid,
            })

    return out


def get_rosters(season: int, league_id: str) -> List[Dict[str, Any]]:
    lg = _league(season, league_id)
    espn_to_canon = _espn_to_canon_cached()

    rosters: List[Dict[str, Any]] = []

    for t in lg.teams or []:
        roster_id = _safe_int(getattr(t, "team_id", None))
        owners = getattr(t, "owners", None) or []
        owner_id = str(owners[0].get("id")) if owners else None

        wins = _safe_int(getattr(t, "wins", None))
        losses = _safe_int(getattr(t, "losses", None))
        ties = _safe_int(getattr(t, "ties", None))
        streak = _streak_from_outcomes(getattr(t, "outcomes", None))

        fpts, fpts_dec = _split_points(getattr(t, "points_for", None))
        fpa, fpa_dec = _split_points(getattr(t, "points_against", None))

        players: List[str] = []
        starters: List[str] = []
        reserve: List[str] = []

        for p in getattr(t, "roster", None) or []:
            pid = getattr(p, "playerId", None)
            if pid is None:
                continue
            cp = canon_pid(str(pid), espn_to_canon)
            if not cp:
                continue

            players.append(cp)
            slot = (
                    getattr(p, "slot_position", None)
                    or getattr(p, "slotPosition", None)
                    or getattr(p, "lineupSlot", None)
            )

            slot = (str(slot).strip().upper() if slot is not None else "")

            if slot in ("IR", "RES"):
                reserve.append(cp)
            elif slot in ("", "BE", "BENCH", "INACTIVE"):
                # IMPORTANT: if slot is missing/unknown, treat as bench
                pass
            else:
                starters.append(cp)

        rosters.append({
            "co_owners": None,
            "keepers": None,
            "league_id": str(league_id),
            "metadata": {"record": f"{wins}-{losses}", "streak": streak},
            "owner_id": owner_id,
            "player_map": None,
            "players": players,
            "reserve": reserve,
            "roster_id": roster_id,
            "settings": {
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "fpts": fpts,
                "fpts_decimal": fpts_dec,
                "fpts_against": fpa,
                "fpts_against_decimal": fpa_dec,
                "ppts": 0,
                "ppts_decimal": 0,
                "total_moves": 0,
                "waiver_budget_used": 0,
                "waiver_position": 0,
            },
            "starters": starters,
            "taxi": None,
        })

    return rosters


# Put this near the top of the module
STARTER_SLOT_ORDER = {
    "QB": 0,
    "RB": 1,
    "WR": 2,
    "TE": 3,
    "RB/WR/TE": 4,
    "FLEX": 4,  # some leagues label it this way
    "OP": 4,  # offensive player flex in some formats
    "K": 5,
    "D/ST": 6,
    "DST": 6,
    "DEF": 6,
}

# ESPN's common proTeamId -> abbrev mapping (covers modern NFL)
ESPN_PROTEAMID_TO_ABBR = {
    0: "FA",
    1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN", 8: "DET",
    9: "GB", 10: "TEN", 11: "IND", 12: "KC", 13: "LV", 14: "LAR", 15: "MIA", 16: "MIN",
    17: "NE", 18: "NO", 19: "NYG", 20: "NYJ", 21: "PHI", 22: "ARI", 23: "PIT", 24: "LAC",
    25: "SF", 26: "SEA", 27: "TB", 28: "WAS", 29: "CAR", 30: "JAX",
    33: "BAL", 34: "HOU",
}


def _norm_slot(slot: Any) -> str:
    s = (str(slot or "").strip().upper())
    if s in ("D/ST", "D-ST", "DEFENSE"):
        return "D/ST"
    if s in ("DST",):
        return "DST"
    if s in ("FLEX", "RB/WR/TE", "RBWRTE"):
        return "RB/WR/TE"
    return s


def _slot_rank(slot: Any) -> int:
    return STARTER_SLOT_ORDER.get(_norm_slot(slot), 999)


def _dst_canonical_id(bp: Any, pid_raw: int) -> Optional[str]:
    # Prefer proTeamId if present
    pro_id = (
            getattr(bp, "proTeamId", None)
            or getattr(bp, "pro_team_id", None)
            or getattr(bp, "proTeam", None)
    )
    if pro_id == "WSH":
        pro_id = "WAS"

    if isinstance(pro_id, str):
        pro_id = None

    # Fallback: derive from ESPN -1600x convention
    if pro_id is None and isinstance(pid_raw, int) and pid_raw < 0:
        pro_id = abs(pid_raw) - 16000  # -16009 -> 9

    try:
        pro_id = int(pro_id) if pro_id is not None else None
    except Exception:
        pro_id = None

    abbr = ESPN_PROTEAMID_TO_ABBR.get(pro_id) if pro_id is not None else None
    if not abbr or abbr == "FA":
        return None
    return abbr


def get_matchups(season: int, league_id: str, week: int) -> List[Dict[str, Any]]:
    espn_to_canon = _espn_to_canon_cached()
    box_scores = _box_scores_cached(season, league_id, week)

    out: List[Dict[str, Any]] = []
    matchup_id = 0

    for bs in box_scores:
        home = getattr(bs, "home_team", None)
        away = getattr(bs, "away_team", None)
        if not home or not away:
            continue  # bye week

        matchup_id += 1

        def build_side(team, lineup, score):
            players: List[str] = []
            players_points: Dict[str, float] = {}

            starter_entries: List[Tuple[int, int, str, float]] = []
            # (rank, original_index, pid, pts) so ties preserve lineup order

            for i, bp in enumerate(lineup or []):
                pid_raw = getattr(bp, "playerId", None) or getattr(bp, "player_id", None)
                if pid_raw is None:
                    continue

                # Canonicalize player id (special-case D/ST)
                cp: Optional[str] = None
                try:
                    pid_int = int(pid_raw)
                except Exception:
                    pid_int = None

                if pid_int is not None and pid_int < 0 and str(pid_int).startswith("-160"):
                    cp = _dst_canonical_id(bp, pid_int)
                else:
                    cp = canon_pid(str(pid_raw), espn_to_canon)

                if not cp:
                    continue

                pts = safe_float(getattr(bp, "points", None))
                slot = getattr(bp, "slot_position", None) or getattr(bp, "slotPosition", None) or getattr(bp,
                                                                                                          "lineupSlot",
                                                                                                          None)

                players.append(cp)
                players_points[cp] = pts

                # starter?
                if slot not in ("BE", "Bench", "IR", "RES", "Inactive"):
                    starter_entries.append((_slot_rank(slot), i, cp, pts))

            starter_entries.sort(key=lambda t: (t[0], t[1]))
            starters = [cp for _, _, cp, _ in starter_entries]
            starters_points = [pts for _, _, _, pts in starter_entries]

            return {
                "points": safe_float(score),
                "players": players,
                "roster_id": _safe_int(getattr(team, "team_id", None) or getattr(team, "id", None)),
                "custom_points": None,
                "matchup_id": matchup_id,
                "starters": starters,
                "starters_points": starters_points,
                "players_points": players_points,
            }

        out.append(build_side(home, getattr(bs, "home_lineup", None), getattr(bs, "home_score", None)))
        out.append(build_side(away, getattr(bs, "away_lineup", None), getattr(bs, "away_score", None)))

    return out


def espn_get_bracket_like(
        league_id: str,
        season: int,
        kind: str,
) -> List[Dict[str, Any]]:
    schedule = _playoff_schedule_cached(season, league_id)
    kind = kind.lower()

    def keep(g):
        p = (g.get("playoffTierType") or "").upper()
        return ("WINNERS" in p) if kind == "winners" else ("LOSERS" in p or "CONSOLATION" in p)

    games = [g for g in schedule if g.get("home") and g.get("away") and keep(g)]
    if not games:
        return []

    rounds = sorted({_safe_int(g.get("matchupPeriodId")) for g in games})
    rmap = {mp: i + 1 for i, mp in enumerate(rounds)}

    out = []
    for g in games:
        h, a = g["home"], g["away"]
        out.append({
            "r": rmap.get(_safe_int(g.get("matchupPeriodId")), 1),
            "m": _safe_int(g.get("id")),
            "t1": _safe_int(h.get("teamId")),
            "t2": _safe_int(a.get("teamId")),
            "t1_from": None,
            "t2_from": None,
            "w": None,
            "l": None,
        })

    return sorted(out, key=lambda x: (x["r"], x["m"]))


@lru_cache(maxsize=32)
def _all_transactions_cached(season: int, league_id: str) -> Dict[int, List[Dict[str, Any]]]:
    """
    Fetch all transactions for a season from ESPN's mTransactions2 view and
    return them keyed by scoring-period (week) number.

    Cached per season+league so the 18 per-week calls in build_week_activity
    only hit the ESPN API once.
    """
    try:
        lg = _league(season, league_id)
        data = lg.espn_request.league_get(params={"view": "mTransactions2"})
    except Exception as exc:
        print(f"[ESPN] _all_transactions_cached failed: {exc}")
        return {}

    raw_txs = data.get("transactions") or []
    espn_to_canon = _espn_to_canon_cached()

    _TYPE_MAP = {
        "WAIVER": "waiver",
        "WAIVER_RESOLUTION": "waiver",
        "FREE_AGENT": "free_agent",
        "TRADE_ACCEPT": "trade",
        "TRADE_ACCEPTED": "trade",
    }

    by_week: Dict[int, List[Dict[str, Any]]] = {}

    for tx in raw_txs:
        tx_type_raw = (tx.get("type") or "").upper()
        tx_type = _TYPE_MAP.get(tx_type_raw)
        if not tx_type:
            continue

        # Only fully-executed transactions
        exec_type = (tx.get("executionType") or "").upper()
        if exec_type not in ("EXECUTE", "EXECUTED", "PROCESS"):
            continue

        scoring_period = int(tx.get("scoringPeriodId") or 0)
        process_ms = tx.get("processDate") or tx.get("proposedDate") or 0

        adds: Dict[str, int] = {}
        drops: Dict[str, int] = {}
        roster_ids: set = set()

        for item in tx.get("items") or []:
            item_type = (item.get("type") or "").upper()
            player_id = item.get("playerId")
            from_team = item.get("fromTeamId")
            to_team = item.get("toTeamId")

            if not player_id or int(player_id) <= 0:
                continue

            cp = canon_pid(str(player_id), espn_to_canon)
            if not cp:
                continue

            if item_type in ("ADDED", "WAIVER_ADDED", "PICKED_UP"):
                if to_team:
                    adds[cp] = int(to_team)
                    roster_ids.add(int(to_team))
            elif item_type in ("DROPPED", "WAIVER_DROPPED"):
                if from_team:
                    drops[cp] = int(from_team)
                    roster_ids.add(int(from_team))
            elif item_type == "TRADED_TO":
                if to_team:
                    adds[cp] = int(to_team)
                    roster_ids.add(int(to_team))
            elif item_type == "TRADED_FROM":
                if from_team:
                    drops[cp] = int(from_team)
                    roster_ids.add(int(from_team))

        if not adds and not drops:
            continue

        entry = {
            "type": tx_type,
            "adds": adds or None,
            "drops": drops or None,
            "roster_ids": sorted(roster_ids),
            "draft_picks": [],
            "status": "complete",
            "created": int(process_ms),
            "status_updated": int(process_ms),
            "leg": scoring_period,
            "transaction_id": str(tx.get("id") or ""),
            "consenter_ids": [],
            "metadata": {},
        }

        by_week.setdefault(scoring_period, []).append(entry)

    return by_week


def get_transactions(season: int, league_id: str, week: int) -> List[Dict[str, Any]]:
    """Return transactions for a specific week, fetching the full season once."""
    return _all_transactions_cached(season, league_id).get(week, [])


def get_drafts(season: int, league_id: str) -> List[Dict[str, Any]]:
    """
    Return a minimal draft record so has_draft_ended() works correctly.
    ESPN seasons always have a completed draft; we use Aug 1 of the season
    year as a conservative start_time so the draft is always treated as ended.
    """
    from datetime import datetime
    start_ts_ms = int(datetime(int(season), 8, 1).timestamp() * 1000)
    return [{
        "draft_id": f"espn_{league_id}_{season}",
        "league_id": str(league_id),
        "season": int(season),
        "season_type": "regular",
        "start_time": start_ts_ms,
        "status": "complete",
        "type": "snake",
    }]


# ESPN slot name -> Sleeper roster position
_ESPN_SLOT_TO_SLEEPER: Dict[str, str] = {
    "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE",
    "FLEX": "FLEX", "RB/WR/TE": "FLEX", "RB/WR": "FLEX",
    "OP": "SUPER_FLEX",
    "K": "K",
    "D/ST": "DEF", "DST": "DEF", "DEF": "DEF", "D-ST": "DEF",
    "BE": "BN", "BENCH": "BN",
    "IR": "IR",
}


def get_league_globals(season: int, league_id: str) -> Dict[str, Any]:
    """
    Extract ESPN league settings in Sleeper-compatible format.
    Returns a dict with: scoring_settings, roster_positions, league_settings, total_rosters.
    Called by platform_api.sync_league_globals() to populate api.py module globals.
    """
    try:
        lg = _league(season, league_id)
    except Exception as e:
        print(f"[espn] get_league_globals failed: {e}")
        return {}

    settings = getattr(lg, "settings", None)

    # Scoring type -> PPR value
    scoring_type = (getattr(settings, "scoring_type", None) or "standard").lower().replace(" ", "_")
    ppr_map = {"ppr": 1.0, "half_ppr": 0.5, "half-ppr": 0.5, "standard": 0.0}
    ppr = ppr_map.get(scoring_type, 0.0)

    scoring_settings: Dict[str, Any] = {
        "rec": ppr,
        "pass_yd": 0.04,
        "pass_td": 4.0,
        "pass_int": -2.0,
        "rush_yd": 0.1,
        "rush_td": 6.0,
        "rec_yd": 0.1,
        "rec_td": 6.0,
        "fum_lost": -2.0,
        "2pt": 2.0,
        "fg_0_19": 3.0,
        "fg_20_29": 3.0,
        "fg_30_39": 3.0,
        "fg_40_49": 4.0,
        "fg_50p": 5.0,
        "xpt": 1.0,
    }

    # Roster positions
    raw_slots = getattr(settings, "roster_slots", None) or []
    roster_positions = [
        _ESPN_SLOT_TO_SLEEPER.get(str(s).upper().strip(), str(s).upper())
        for s in raw_slots
    ]

    # League settings
    total_rosters = len(getattr(lg, "teams", None) or [])
    league_settings: Dict[str, Any] = {
        "playoff_teams": _safe_int(getattr(settings, "playoff_team_count", 4)),
        "num_teams": total_rosters,
        "type": 0,
    }

    return {
        "scoring_settings": scoring_settings,
        "roster_positions": roster_positions,
        "league_settings": league_settings,
        "total_rosters": total_rosters,
    }


def build_espn_to_canonical(players_index: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """
    returns: espnId(str) -> canonical_id(str) where canonical_id is the dict key in your index
    Example index:
      "5938": {..., "espnID": "4039057"}
    => map["4039057"] = "5938"
    """
    out: Dict[str, str] = {}
    for canonical_id, info in (players_index or {}).items():
        espn_id = info.get("espnID") or info.get("espnId") or info.get("espn_id")
        if espn_id:
            out[str(espn_id)] = str(canonical_id)
    return out


def canon_pid(espn_pid: Any, espn_to_canon: Dict[str, str]) -> Optional[str]:
    if espn_pid is None:
        return None
    return espn_to_canon.get(str(espn_pid))
