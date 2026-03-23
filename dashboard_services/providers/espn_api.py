# dashboard_services/espn_api.py
from __future__ import annotations

import os
from espn_api.football import League
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

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


def _safe_float(x: Any, default: float = 0.0) -> float:
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
    f = _safe_float(val, 0.0)
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
    return League(
        league_id=int(league_id),
        year=int(season),
        espn_s2=_require_env("ESPN_S2"),
        swid=_normalize_swid(_require_env("ESPN_SWID")),
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

                pts = _safe_float(getattr(bp, "points", None))
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
                "points": _safe_float(score),
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


def get_transactions(season: int, league_id: str, week: int) -> List[Dict[str, Any]]:
    return []


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
