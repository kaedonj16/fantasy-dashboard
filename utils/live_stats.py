"""Consolidated utils module: live_stats.

Live stats normalization and Tank01-backed week-stats builders (split from utils/utils.py).
"""

# --- imports carried over from utils/utils.py ---
import glob
import json
import os
import re
import threading as _threading
import requests
import time
import traceback
import uuid
from contextlib import contextmanager as _contextmanager
from bs4 import BeautifulSoup
from collections import OrderedDict as _OrderedDict, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any, Callable, List, Iterable, TYPE_CHECKING
from dashboard_services.api import (
    _fetch_league,
    get_nfl_games_for_week_raw,
    get_transactions,
    get_rosters,
    get_users,
    get_traded_picks,
    get_nfl_state,
    get_nfl_players, fetch_team_game_logs_html, fetch_tank_boxscore, get_matchups,
)
from dashboard_services.display_names import team_label_from_user, username_from_user
from utils.projections import TANK01_API_HOST, TANK01_API_KEY, _headers

# ======================================================================
# From utils/utils.py (split per consolidation map)
# ======================================================================

# --- utils/utils.py L2017 ---
def _safe_int(val: Any) -> int:
    if val is None:
        return 0
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val).strip()
    if not s:
        return 0
    try:
        return int(float(s))
    except ValueError:
        return 0

# --- utils/utils.py L2031 ---
def build_tank_player_index(players_index: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for sleeper_id, meta in players_index.items():
        tank_id = meta.get("tankId")
        if not tank_id:
            continue
        out[str(tank_id)] = {
            "name": meta.get("name", ""),
            "team": meta.get("team", ""),
            "pos": meta.get("pos", ""),
        }
    return out

# --- utils/utils.py L2045 ---
def _normalize_qb_stats(ps: Dict[str, Any]) -> Dict[str, int]:
    passing = ps.get("Passing") or {}
    rushing = ps.get("Rushing") or {}
    return {
        "pass_yds": _safe_int(passing.get("passYds")),
        "pass_td": _safe_int(passing.get("passTD")),
        "int": _safe_int(passing.get("int")),
        "rush_att": _safe_int(rushing.get("carries")),
        "rush_yds": _safe_int(rushing.get("rushYds")),
        "rush_td": _safe_int(rushing.get("rushTD")),
    }

# --- utils/utils.py L2058 ---
def _normalize_skill_stats(ps: Dict[str, Any]) -> Dict[str, int]:
    rushing = ps.get("Rushing") or {}
    receiving = ps.get("Receiving") or {}
    return {
        "rush_att": _safe_int(rushing.get("carries")),
        "rush_yds": _safe_int(rushing.get("rushYds")),
        "rush_td": _safe_int(rushing.get("rushTD")),
        "rec": _safe_int(receiving.get("receptions")),
        "rec_yds": _safe_int(receiving.get("recYds")),
        "rec_td": _safe_int(receiving.get("recTD")),
    }

# --- utils/utils.py L2071 ---
def _normalize_dst_stats(team_block: Dict[str, Any]) -> Dict[str, int]:
    """
    Normalize team DST block into DEF stats.
    Example fields from Tank01:
      {
        "teamAbv": "WSH",
        "defTD": "0",
        "defensiveInterceptions": "0",
        "sacks": "4",
        "ydsAllowed": "313",
        "fumblesRecovered": "0",
        "ptsAllowed": "31",
        "safeties": "0"
      }
    """
    return {
        "def_td": _safe_int(team_block.get("defTD")),
        "def_int": _safe_int(team_block.get("defensiveInterceptions")),
        "sacks": _safe_int(team_block.get("sacks")),
        "yds_allowed": _safe_int(team_block.get("ydsAllowed")),
        "fumbles_recovered": _safe_int(team_block.get("fumblesRecovered")),
        "pts_allowed": _safe_int(team_block.get("ptsAllowed")),
        "safeties": _safe_int(team_block.get("safeties")),
    }

# --- utils/utils.py L2097 ---
def build_live_stats_for_game_from_tank(
        boxscore: dict,
        players_index: dict,
) -> dict:
    """
    Returns:
      {
        "MIN": {
          "QB": {...},
          "RB": {...},
          "WR": {...},
          "TE": {...},
          "DEF": { "team": {...} }
        },
        "WSH": { ... }
      }
    """

    tank_idx = build_tank_player_index(players_index)

    # Note DEF is included here
    out: dict[str, dict[str, dict[str, dict]]] = defaultdict(
        lambda: {"QB": {}, "RB": {}, "WR": {}, "TE": {}, "DEF": {}}
    )

    # --- Player-level offense stats ---
    player_stats = boxscore.get("playerStats") or {}
    if isinstance(player_stats, dict):
        for tank_id, ps in player_stats.items():
            meta = tank_idx.get(str(tank_id))
            if not meta:
                continue

            team = meta.get("team") or ps.get("teamAbv") or ps.get("team")
            pos = meta.get("pos")
            name_key = (meta.get("name") or ps.get("longName") or "").lower()

            if not team or not pos or not name_key:
                continue

            if pos == "QB":
                stats = _normalize_qb_stats(ps)
                out[team]["QB"][name_key] = stats
            elif pos in ("RB", "WR", "TE"):
                stats = _normalize_skill_stats(ps)
                out[team][pos][name_key] = stats
            else:
                # still ignoring K/IDP; handled via team DST for DEF
                continue

    # --- Team DST → DEF (named as "DEF") ---
    dst_block = boxscore.get("DST") or {}
    if isinstance(dst_block, dict):
        for side in ("home", "away"):
            team_block = dst_block.get(side)
            if not isinstance(team_block, dict):
                continue
            team_abv = team_block.get("teamAbv") or team_block.get("team")
            if not team_abv:
                continue

            def_stats = _normalize_dst_stats(team_block)
            # Put under DEF, with a "team" key
            out[team_abv]["DEF"]["team"] = def_stats

    return out

# --- utils/utils.py L2165 ---
def merge_live_stats_into_league_week_stats(
        league_week_stats: dict,
        live_stats: dict,
) -> None:
    """
    Mutates league_week_stats in-place, overwriting or adding stats
    for any players that appear in live_stats.
    Shape of both dicts:

      league_week_stats[team][pos][player_name] = {stat_dict}

    Marks each merged player line with ``_src: "tank"`` so matchup rows can
    tell Tank01 box scores apart from Footballguys prior-season leftovers.
    """
    for team, pos_map in live_stats.items():
        team_bucket = league_week_stats.setdefault(team, {})
        for pos, players in pos_map.items():
            pos_bucket = team_bucket.setdefault(pos, {})
            for name_key, stat_dict in players.items():
                if isinstance(stat_dict, dict):
                    merged = dict(stat_dict)
                    merged["_src"] = "tank"
                    pos_bucket[name_key] = merged
                else:
                    pos_bucket[name_key] = stat_dict

# --- utils/utils.py L2192 ---
def get_live_game_ids_for_today(
        schedule: Iterable[Dict[str, Any]],
        today: date | None = None,
) -> List[str]:
    """
    Return Tank01 gameIDs for games:
      - with gameDate = today's date
      - AND whose epoch time indicates the game is currently within
        the live window (kickoff to +3 hours)
    """
    if today is None:
        today = date.today()

    today_str = today.strftime("%Y%m%d")
    live_ids: List[str] = []

    now = time.time()
    three_hours = 4 * 60 * 60  # 10800 seconds

    for game in schedule:
        if not isinstance(game, dict):
            continue

        # Must match today's date
        if str(game.get("gameDate") or "") != today_str:
            continue

        # Must have an epoch value
        raw_epoch = game.get("gameTime_epoch")
        if raw_epoch is None:
            continue

        try:
            game_epoch = float(raw_epoch)
        except (ValueError, TypeError):
            continue

        if game_epoch <= now <= (game_epoch + three_hours):
            gid = game.get("gameID")
            if gid:
                live_ids.append(str(gid))

    return live_ids if live_ids else []

# --- utils/utils.py L2237 ---
def get_started_game_ids_for_week(
        schedule: Iterable[Dict[str, Any]],
        now: datetime | None = None,
) -> List[str]:
    """Tank01 gameIDs for games that have kicked off (live or final).

    Includes calendar-past rows even when ``gameStatusCode`` is still ``0``,
    so Thursday finals keep getting boxscore overlays after the live window.
    """
    from utils.game_status import game_has_started
    ids: List[str] = []
    seen: set[str] = set()
    for game in schedule or []:
        if not isinstance(game, dict):
            continue
        if not game_has_started(game, now=now):
            continue
        gid = game.get("gameID")
        if not gid:
            continue
        key = str(gid)
        if key in seen:
            continue
        seen.add(key)
        ids.append(key)
    return ids

# --- utils/utils.py L2264 ---
def tank01_status_confirms_started(game: Optional[dict]) -> bool:
    """True when Tank01 itself reports live/final (not merely a past calendar day)."""
    if not game or not isinstance(game, dict):
        return False
    return str(game.get("gameStatusCode") or "").strip() in ("1", "2")

# --- utils/utils.py L2271 ---
def player_week_stat_entry(
        teams_stats: Optional[Dict[str, Any]],
        team: str,
        pos: str,
        player: str,
) -> Optional[Dict[str, Any]]:
    """Raw week_stats dict for one player (includes optional ``_src``)."""
    from utils.players import normalize_name
    from utils.nfl import lookup_team_map
    if not teams_stats or not team or not player:
        return None
    pos_norm = (pos or "").strip().upper()
    if pos_norm == "PK":
        lookup_pos = "K"
    elif pos_norm in ("DEF", "DST", "D/ST"):
        lookup_pos = "DEF"
    else:
        defensive_positions = {
            "DL", "DE", "DT", "EDGE", "LB", "ILB", "OLB",
            "DB", "CB", "S", "FS", "SS", "IDP",
        }
        lookup_pos = "IDP" if pos_norm in defensive_positions else pos_norm
    team_data = lookup_team_map(teams_stats, team) or {}
    if lookup_pos == "DEF":
        return None
    pos_data = team_data.get(lookup_pos) or {}
    if not isinstance(pos_data, dict):
        return None
    wanted = normalize_name(player)

    def find_in(bucket):
        from utils.players import normalize_name
        if not isinstance(bucket, dict):
            return None
        direct = bucket.get(wanted)
        if isinstance(direct, dict):
            return direct
        # Feeds disagree on common given-name forms (Kenneth/Ken) while the
        # canonical surname and first stem remain stable. Only accept a unique
        # candidate so namesakes can never acquire one another's box score.
        parts = wanted.split()
        if len(parts) >= 2:
            matches = []
            for raw_name, value in bucket.items():
                candidate = normalize_name(raw_name).split()
                if (isinstance(value, dict) and len(candidate) >= 2
                        and candidate[-1] == parts[-1]
                        and candidate[0][:3] == parts[0][:3]):
                    matches.append(value)
            if len(matches) == 1:
                return matches[0]
        return None

    entry = find_in(pos_data)
    if entry is not None:
        return entry

    # Historical stats belong to the team the player represented that week,
    # not necessarily the current team in the player index. Search other team
    # buckets only when the identity match is unique across the weekly snapshot.
    matches = []
    for other_team in (teams_stats or {}).values():
        if not isinstance(other_team, dict):
            continue
        found = find_in(other_team.get(lookup_pos) or {})
        if found is not None and all(found is not existing for existing in matches):
            matches.append(found)
    return matches[0] if len(matches) == 1 else None

# --- utils/utils.py L2338 ---
def box_score_line_is_trusted(
        game: Optional[dict],
        player_stats: Optional[dict],
) -> bool:
    """Whether a week_stats line is safe to show on matchup rows.

    Footballguys republishes last year's Wk N under the new season until their
    logs flip. Calendar-past + Tank code ``0`` used to unlock those leftovers.
    Trust Tank-overlaid lines (``_src=tank``) or an explicit Tank live/final code.
    """
    if player_stats and player_stats.get("_src") == "tank":
        return True
    return tank01_status_confirms_started(game)

# --- utils/utils.py L2357 ---
def sleeper_week_stats_path(season: int, week: int) -> Optional[str]:
    """Path to the Sleeper per-player stats file for one NFL week, or None.

    Both the live in-season fetcher and the history backfill write the
    non-dated ``sleeper_stats_s{Y}_w{W}.json`` into cache/sleeper_stats/; some
    backfills also leave a dated ``..._w{W}_{date}.json``. Prefer the freshest.
    """
    from utils.data_cache import CACHE_DIR
    stats_dir = CACHE_DIR / "sleeper_stats"
    candidates = list(glob.glob(str(stats_dir / f"sleeper_stats_s{int(season)}_w{int(week)}_*.json")))
    non_dated = stats_dir / f"sleeper_stats_s{int(season)}_w{int(week)}.json"
    if non_dated.exists():
        candidates.append(str(non_dated))
    if not candidates:
        legacy = CACHE_DIR / f"sleeper_stats_s{int(season)}_w{int(week)}.json"
        return str(legacy) if legacy.exists() else None
    return max(candidates, key=os.path.getmtime)

# --- utils/utils.py L2375 ---
def load_sleeper_week_stats(season: int, week: int) -> Dict[str, Any]:
    """Per-player-id Sleeper stat lines for one NFL week (cached, read-only).

    This is the same cache family the player-modal game log reads, so it has a
    line for every player who played -- unlike the Footballguys team scrape,
    which can miss rookies and mid-week adds. Returns ``{}`` when absent.
    """
    from utils.data_cache import read_json_cached
    path = sleeper_week_stats_path(season, week)
    if not path:
        return {}
    data = read_json_cached(path)
    return data if isinstance(data, dict) else {}

# --- utils/utils.py L2389 ---
def overlay_idp_and_k_stats_from_sleeper(
        league_week_stats: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
        season: int,
        week: int,
        teams_index: Dict[str, Dict[str, Any]],
) -> None:
    """
    Mutates league_week_stats in place by adding:
      - IDP stats under:  league_week_stats[TEAM]["IDP"][name_lower] = {...}
      - K stats under:    league_week_stats[TEAM]["K"][name_lower]   = {...}

    Normalizes team abbreviations (e.g., WSH -> WAS) so keys match teams_index.
    """
    from utils.data_cache import read_json_cached
    from utils.data_cache import load_players_index
    from utils.data_cache import load_idp_index

    # ----- Load IDP index -----
    idp_index = load_idp_index()
    if not isinstance(idp_index, dict):
        print("[week_stats][IDP/K] idp_players_index.json is not a dict, skipping.")
        return

    # ----- Load main player index for kickers -----
    players_index = load_players_index()
    if not isinstance(players_index, dict):
        players_index = {}
        print("[week_stats][IDP/K] players_index not available; kicker overlay may be skipped.")

    # ----- Find the Sleeper stats file for this season/week -----
    # The overlay runs at build time and lazily at render time for legacy weekly
    # snapshots, so use the shared, mtime-cached loader (it matches both the
    # non-dated live file and any dated backfill).
    sleeper_stats_path = sleeper_week_stats_path(season, week)
    if not sleeper_stats_path:
        print(
            f"[week_stats][IDP/K] No sleeper stats file for season={season} week={week}"
        )
        return
    print(f"[week_stats][IDP/K] Using Sleeper stats file: {os.path.basename(sleeper_stats_path)}")

    sleeper_stats = read_json_cached(sleeper_stats_path)
    if not isinstance(sleeper_stats, dict):
        print("[week_stats][IDP/K] Sleeper stats JSON missing or not a dict, skipping.")
        return

    valid_teams = set((teams_index or {}).keys())

    def normalize_team_abv(abv: str) -> str:
        from utils.nfl import TEAM_ALIASES
        abv = (abv or "").strip().upper()
        abv = TEAM_ALIASES.get(abv, abv)

        # If teams_index uses the opposite code (rare), still land on a valid key
        if abv and abv not in valid_teams:
            for k, v in TEAM_ALIASES.items():
                if v == abv and k in valid_teams:
                    return k
        return abv

    def clean_numeric_stats(raw_stats: Any) -> Dict[str, float]:
        if not isinstance(raw_stats, dict):
            return {}
        out: Dict[str, float] = {}
        for k, v in raw_stats.items():
            if isinstance(v, (int, float)):
                out[k] = float(v)
        return out

    IDP_BUCKET = "IDP"
    K_BUCKET = "K"

    idp_added = 0
    k_added = 0

    for sleeper_id, raw_stats in sleeper_stats.items():
        stats_clean = clean_numeric_stats(raw_stats)
        if not stats_clean:
            continue

        sid = str(sleeper_id)

        # --------------------
        # IDP overlay
        # --------------------
        meta_idp = idp_index.get(sid)
        if meta_idp:
            name = (meta_idp.get("name") or "").strip()
            team_abv = normalize_team_abv(meta_idp.get("team") or "")
            pos = (meta_idp.get("pos") or "").strip().upper()  # DB/DL/LB

            if not name or not team_abv:
                continue
            if valid_teams and team_abv not in valid_teams:
                continue

            name_key = name.lower()
            stats_clean_idp = dict(stats_clean)
            stats_clean_idp.setdefault("pos", pos)

            team_bucket = league_week_stats.setdefault(team_abv, {})
            idp_bucket = team_bucket.setdefault(IDP_BUCKET, {})

            existing = idp_bucket.get(name_key, {})
            if isinstance(existing, dict):
                existing.update(stats_clean_idp)
                idp_bucket[name_key] = existing
            else:
                idp_bucket[name_key] = stats_clean_idp

            idp_added += 1

            # IMPORTANT: if it’s IDP, don’t also treat as kicker
            continue

        # --------------------
        # K overlay (from players_index)
        # --------------------
        meta_p = players_index.get(sid)
        if not meta_p:
            continue

        pos = (meta_p.get("pos") or "").strip().upper()
        if pos != "PK":
            continue

        name = (meta_p.get("name") or "").strip()
        team_abv = normalize_team_abv(meta_p.get("team") or "")

        if not name or not team_abv:
            continue
        if valid_teams and team_abv not in valid_teams:
            continue

        name_key = name.lower()
        stats_clean_k = dict(stats_clean)
        stats_clean_k.setdefault("pos", "PK")

        team_bucket = league_week_stats.setdefault(team_abv, {})
        k_bucket = team_bucket.setdefault(K_BUCKET, {})

        existing = k_bucket.get(name_key, {})
        if isinstance(existing, dict):
            existing.update(stats_clean_k)
            k_bucket[name_key] = existing
        else:
            k_bucket[name_key] = stats_clean_k

        k_added += 1

    # Optional cleanup: if something else already created WSH, merge into WAS and delete WSH
    if "WSH" in league_week_stats and "WAS" in league_week_stats:
        try:
            for bucket, players in (league_week_stats.get("WSH") or {}).items():
                dest_bucket = league_week_stats["WAS"].setdefault(bucket, {})
                if isinstance(players, dict):
                    for name_key, stat_blob in players.items():
                        if isinstance(stat_blob, dict):
                            dest_bucket.setdefault(name_key, {}).update(stat_blob)
                        else:
                            dest_bucket[name_key] = stat_blob
            del league_week_stats["WSH"]
        except Exception:
            # don’t break the pipeline for a cleanup step
            pass

    print(f"[week_stats][IDP/K] Added/updated {idp_added} IDP stat lines.")
    print(f"[week_stats][IDP/K] Added/updated {k_added} K stat lines.")

# --- utils/utils.py L2563 ---
def build_and_save_week_stats_for_league(
        teams_index: Dict[str, Dict[str, Any]],
        season: int,
        week: int,
        live_game_ids: Optional[Iterable[str]] = None,
) -> Path:
    """
    1) Build baseline week stats from your existing HTML pipeline.
    2) Optionally overlay live Tank01 stats for any provided gameIDs.
    3) Overlay IDP stats from Sleeper using idp_players_index + sleeper_stats file.
    """
    from utils.data_cache import write_json
    from utils.data_cache import path_week_stats
    from utils.data_cache import load_week_stats
    from utils.data_cache import load_week_schedule
    from utils.data_cache import load_players_index
    league_week_stats: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    # Footballguys "Wk N" still holds last season's box scores until this
    # week's games are actually played. Scraping that into week_stats_s{year}
    # makes the Weekly Hub look like players already suited up after a draft.
    schedule: List[Dict[str, Any]] = []
    try:
        raw_sched = load_week_schedule(season, week)
        if isinstance(raw_sched, list):
            schedule = raw_sched
        elif isinstance(raw_sched, dict):
            maybe = raw_sched.get("body") or raw_sched.get("games") or []
            if isinstance(maybe, list):
                schedule = maybe
    except Exception:
        schedule = []

    # Calendar-past + Tank code 0 means the game is over but Footballguys may
    # still be republishing last year. Only scrape FG once Tank itself reports
    # live/final (or we already have explicit live IDs). Always Tank-overlay
    # every started game so Thursday finals don't fall back to FG leftovers.
    tank_confirmed = any(
        tank01_status_confirms_started(g) for g in schedule if isinstance(g, dict)
    )
    started_ids = get_started_game_ids_for_week(schedule)
    overlay_ids: List[str] = []
    seen_ids: set[str] = set()
    for gid in list(live_game_ids or []) + started_ids:
        key = str(gid)
        if not key or key in seen_ids:
            continue
        seen_ids.add(key)
        overlay_ids.append(key)

    allow_fg_scrape = tank_confirmed or bool(live_game_ids)
    if not allow_fg_scrape and not overlay_ids:
        out_path = path_week_stats(season, week)
        if schedule:
            write_json(out_path, {})
            print(f"[week_stats] no games started yet (season={season}, week={week}); wrote empty")
        else:
            print(f"[week_stats] skip scrape; no schedule and no live games (season={season}, week={week})")
        return out_path

    print(f"[week_stats] building stats for the week (season={season}, week={week})")
    if allow_fg_scrape:
        for team_abv in teams_index.keys():
            orig_team_abv = team_abv
            if team_abv == "WSH":
                team_abv = "WAS"
            try:
                html = fetch_team_game_logs_html(team_abv, season)
                pos_player_stats = parse_team_week_pos_player_stats(html, week)
                league_week_stats[team_abv] = pos_player_stats
            except Exception as e:
                print(f"[week_stats] Error for {orig_team_abv} week {week}: {e}")
                league_week_stats[team_abv] = {}
    else:
        # Preserve any prior Tank-backed lines while we refresh overlays.
        prior = load_week_stats(season, week)
        if isinstance(prior, dict):
            league_week_stats = prior

    # ---------- Overlay Tank01 stats for started / live games ----------
    if overlay_ids:
        # Load players_index once, reuse
        players_index = load_players_index()

        session = requests.Session()

        print(f"[week_stats] fetching Tank01 box scores for {len(overlay_ids)} game(s)")
        for game_id in overlay_ids:
            try:
                boxscore = fetch_tank_boxscore(game_id, session=session)
                if not boxscore:
                    continue
                live_stats = build_live_stats_for_game_from_tank(boxscore, players_index)
                merge_live_stats_into_league_week_stats(league_week_stats, live_stats)
            except Exception as e:
                print(f"[week_stats] Tank01 error for {game_id}: {e}")

    # ---------- Overlay IDP stats from Sleeper ----------
    overlay_idp_and_k_stats_from_sleeper(
        league_week_stats=league_week_stats,
        season=season,
        week=week,
        teams_index=teams_index,
    )

    out_path = path_week_stats(season, week)
    write_json(out_path, league_week_stats)
    print(f"[week_stats] Wrote → {out_path}")
    return out_path

# --- utils/utils.py L2693 ---
def parse_team_week_pos_player_stats(
        html: str,
        target_week: int,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    soup = BeautifulSoup(html, "html.parser")

    heading_to_pos = {
        "Quarterbacks": "QB",
        "Running Backs": "RB",
        "Wide Receivers": "WR",
        "Tight Ends": "TE",
    }

    result: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for h2 in soup.find_all("h2"):
        heading = h2.get_text(strip=True)
        pos_code = heading_to_pos.get(heading)
        if not pos_code:
            continue

        table = h2.find_next("table")
        if not table:
            continue

        pos_players = parse_position_table_for_week(table, pos_code, target_week)
        result[pos_code] = pos_players

    return result

# --- utils/utils.py L2724 ---
def parse_position_table_for_week(
        table,
        pos_code: str,
        target_week: int,
) -> Dict[str, Dict[str, Any]]:
    from utils.players import normalize_name
    thead = table.find("thead")
    tbody = table.find("tbody")
    if not thead or not tbody:
        return {}

    head_rows = thead.find_all("tr")
    if not head_rows:
        return {}

    week_hdr_cells = head_rows[0].find_all("th")
    week_col_idx = None
    target_label = f"Wk {target_week}".lower()

    for i, th in enumerate(week_hdr_cells):
        txt = th.get_text(strip=True).lower()
        if txt == target_label:
            week_col_idx = i
            break

    if week_col_idx is None:
        return {}

    pos_players: Dict[str, Dict[str, Any]] = {}

    for tr in tbody.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) <= week_col_idx:
            continue

        name_cell = cells[0]
        link = name_cell.find("a")
        player_name = (
            link.get_text(strip=True) if link is not None else name_cell.get_text(strip=True)
        )
        if not player_name:
            continue
        player_name = normalize_name(player_name)

        stat_cell = cells[week_col_idx]
        plain_text = stat_cell.get_text(strip=True)
        if plain_text == "" or plain_text == "0":
            continue

        lines = list(stat_cell.stripped_strings)
        stats = parse_stat_lines_for_pos(lines, pos_code)
        if stats:
            pos_players[player_name] = stats

    return pos_players

# --- utils/utils.py L2780 ---
def parse_stat_lines_for_pos(lines, pos_code: str) -> Dict[str, Any]:
    """
    lines: list of text lines from the cell for that week, e.g.
      QB: ["320-2-1", "3-19-0"]
      RB: ["8-69-0", "1-6-0"]
    """

    def _parse_nums(line: str) -> list[int]:
        line = line.strip()
        if not line:
            return []

        parts = line.split("-")
        nums: list[int] = []
        i = 0
        while i < len(parts):
            part = parts[i].strip()
            if part == "":
                if i + 1 < len(parts) and parts[i + 1].strip():
                    nums.append(-int(parts[i + 1].strip()))
                    i += 2
                else:
                    i += 1
            else:
                nums.append(int(part))
                i += 1
            if len(nums) >= 3:
                break
        return nums

    nums_by_line: list[list[int]] = []
    for line in lines:
        nums = _parse_nums(line)
        if nums:
            nums_by_line.append(nums)

    stats: Dict[str, Any] = {}

    if pos_code == "QB":
        if len(nums_by_line) >= 1 and len(nums_by_line[0]) >= 3:
            py, ptd, ints = nums_by_line[0][:3]
            stats.update({"pass_yds": py, "pass_td": ptd, "int": ints})
        if len(nums_by_line) >= 2 and len(nums_by_line[1]) >= 3:
            ra, ry, rtd = nums_by_line[1][:3]
            stats.update({"rush_att": ra, "rush_yds": ry, "rush_td": rtd})

    elif pos_code in {"RB", "WR"}:
        if len(nums_by_line) >= 1 and len(nums_by_line[0]) >= 3:
            ra, ry, rtd = nums_by_line[0][:3]
            stats.update({"rush_att": ra, "rush_yds": ry, "rush_td": rtd})
        if len(nums_by_line) >= 2 and len(nums_by_line[1]) >= 3:
            rec, r_yards, rtd2 = nums_by_line[1][:3]
            stats.update({"rec": rec, "rec_yds": r_yards, "rec_td": rtd2})

    elif pos_code == "TE":
        if len(nums_by_line) >= 1 and len(nums_by_line[0]) >= 3:
            rec, r_yards, rtd = nums_by_line[0][:3]
            stats.update({"rec": rec, "rec_yds": r_yards, "rec_td": rtd})

    else:
        if len(nums_by_line) >= 1 and len(nums_by_line[0]) >= 3:
            a, b, c = nums_by_line[0][:3]
            stats.update({"stat1": a, "stat2": b, "stat3": c})

    return stats

# --- utils/utils.py L2847 ---
def count_roster_positions(positions: list[str]) -> dict[str, int]:
    """
    Takes a Sleeper roster_positions array and returns a count of each slot type.
    Example:
      ['QB','RB','RB','WR','WR','TE','FLEX',...] → {'QB':1,'RB':2,'WR':2,...}

    Provider aliases (OP, RB/WR/TE, WRRB_FLEX, D/ST, …) collapse to the canonical name.
    Restricted flex (WR/RB only, WR/TE, RB/TE) stays distinct from standard FLEX.
    so Fleaflicker/Yahoo/ESPN slot lists count the same as Sleeper.
    """
    from utils.lineup_slots import count_lineup_slots
    return count_lineup_slots(positions)
