"""Consolidated utils module: data_cache.

paths, JSON IO, table loaders, week-projection cache

Merged from: utils/paths.py, utils/pipeline_health.py.
Old import paths keep working via compatibility shims.
"""
from __future__ import annotations


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


# ======================================================================
# From utils/paths.py
# ======================================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = ROOT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PLAYER_HISTORY_DIR = CACHE_DIR / "player_history"
PLAYER_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

PLAYER_INVESTMENT_DIR = CACHE_DIR / "player_investment"
PLAYER_INVESTMENT_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# From utils/pipeline_health.py
# ======================================================================

"""Shared pipeline-health persistence for cron_daily and the web app.

The cron container's disk is invisible to the web container on Render, so
``cron_daily`` POSTs each step's status to the CRON_SECRET-authenticated
``/api/cron/pipeline-health`` web endpoint, and the web process persists it
with :func:`write_step_health` into its own ``CACHE_DIR/pipeline_health.json``,
which ``/api/health/pipeline`` reads back.
"""

HEALTH_FILENAME = "pipeline_health.json"

_VALID_STATUSES = ("ok", "error", "timeout", "skipped")


def health_path(cache_dir: Path | None = None) -> Path:
    return (cache_dir or CACHE_DIR) / HEALTH_FILENAME


def write_step_health(
    step_name: str,
    status: str,
    at: str | None = None,
    cache_dir: Path | None = None,
) -> dict:
    """Merge one step's status into pipeline_health.json; return the full payload."""
    dest = health_path(cache_dir)
    data: dict = {}
    try:
        if dest.exists():
            data = json.loads(dest.read_text(encoding="utf-8")) or {}
    except Exception:
        data = {}
    now = at or datetime.now(timezone.utc).isoformat()
    entry: dict = {"status": str(status), "at": now}
    if status == "ok":
        entry["last_success"] = now
    else:
        # Keep the previous last_success so "last succeeded at" survives errors.
        prev = data.get(str(step_name)) or {}
        if isinstance(prev, dict) and prev.get("last_success"):
            entry["last_success"] = prev["last_success"]
    data[str(step_name)] = entry
    data["_updated"] = now
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def read_health(cache_dir: Path | None = None) -> dict:
    """Read the pipeline-health payload; {} when missing or unreadable."""
    dest = health_path(cache_dir)
    try:
        if dest.exists():
            data = json.loads(dest.read_text(encoding="utf-8")) or {}
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


# ======================================================================
# From utils/utils.py (split per consolidation map)
# ======================================================================






# --- utils/utils.py L88 ---
STATUS_NOT_STARTED = "not_started"

# --- utils/utils.py L89 ---
STATUS_IN_PROGRESS = "in_progress"

# --- utils/utils.py L90 ---
STATUS_FINAL = "final"

# --- utils/utils.py L264 ---
def path_week_schedule(season: int, week: int) -> str:
    return os.path.join(CACHE_DIR, f"schedule/schedule_s{season}_w{week}.json")

# --- utils/utils.py L268 ---
def path_players_index() -> str:
    return os.path.join(CACHE_DIR, "players_index.json")

# --- utils/utils.py L272 ---
def path_relevant_index() -> str:
    return os.path.join(CACHE_DIR, "players_index_relevant.json")

# --- utils/utils.py L276 ---
def path_usage_table() -> str:
    return os.path.join(DATA_DIR, "usage_table.json")

# --- utils/utils.py L280 ---
def path_engine_table() -> str:
    return os.path.join(DATA_DIR, "engine_values.csv")

# --- utils/utils.py L284 ---
def path_model_value_table() -> str:
    return str(DATA_DIR / "model_values.json")

# --- utils/utils.py L288 ---
def path_teams_index() -> str:
    return os.path.join(CACHE_DIR, "teams_index.json")

# --- utils/utils.py L292 ---
def path_idp_index() -> str:
    return os.path.join(CACHE_DIR, "idp_players_index.json")

# --- utils/utils.py L296 ---
def path_week_proj(season: int, week: int) -> str:
    return os.path.join(CACHE_DIR, f"projections/projections_s{season}_w{week}.json")

# --- utils/utils.py L300 ---
def path_week_stats(season: int, week: int) -> str:
    return os.path.join(CACHE_DIR, f"stats/week_stats_s{season}_w{week}.json")

# --- utils/utils.py L304 ---
def path_fantasycalc_values() -> str:
    return os.path.join(DATA_DIR, "fantasycalc_api_values.csv")

# --- utils/utils.py L308 ---
def path_fantasycalc_sf_values() -> str:
    return os.path.join(DATA_DIR, "fantasycalc_sf_api_values.csv")

# --- utils/utils.py L312 ---
def path_dynastyprocess_values() -> str:
    return os.path.join(DATA_DIR, "dynastyprocess_values.csv")

# --- utils/utils.py L320 ---
def read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# --- utils/utils.py L332 ---
try:
    _JSON_CACHE_MAX = max(1, int(os.getenv("JSON_CACHE_MAX", "16")))
except (TypeError, ValueError):
    _JSON_CACHE_MAX = 16

# --- utils/utils.py L336 ---
_JSON_CACHE: Dict[str, tuple] = _OrderedDict()

# --- utils/utils.py L337 ---
_JSON_CACHE_LOCK = _threading.Lock()

# --- utils/utils.py L340 ---
def read_json_cached(path: str) -> Optional[dict]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    sig = (st.st_mtime_ns, st.st_size)
    with _JSON_CACHE_LOCK:
        hit = _JSON_CACHE.get(path)
        if hit is not None and hit[0] == sig:
            _JSON_CACHE.move_to_end(path)
            return hit[1]
        if hit is not None:
            _JSON_CACHE.pop(path, None)
    data = read_json(path)
    if data is not None:
        with _JSON_CACHE_LOCK:
            _JSON_CACHE[path] = (sig, data)
            _JSON_CACHE.move_to_end(path)
            while len(_JSON_CACHE) > _JSON_CACHE_MAX:
                _JSON_CACHE.popitem(last=False)
    return data

# --- utils/utils.py L363 ---
def write_json(path, data):
    """
    Safely writes a JSON object to disk.
    Works with both string and Path objects.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Unique tmp per writer: a shared tmp name made concurrent writers for
    # the same path collide -- the first replace won and the second raised
    # Errno 2, discarding the work and forcing a refetch.
    tmp = p.parent / (p.name + f".tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(p)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise

# --- utils/utils.py L389 ---
_PLAYERS_INDEX_MERGED: Dict[str, object] = {"sig": None, "data": None}

# --- utils/utils.py L390 ---
_RELEVANT_INDEX_MERGED: Dict[str, object] = {"sig": None, "data": None}

# --- utils/utils.py L393 ---
def _bye_lookup_for_overlay() -> Dict[str, int]:
    """Best-effort team -> byeWeek map for overlay (avoids circular import cost)."""
    try:
        teams = read_json_cached(path_teams_index()) or {}
    except Exception:
        return {}
    out: Dict[str, int] = {}
    for abv, meta in teams.items():
        if not isinstance(meta, dict):
            continue
        bye = meta.get("byeWeek")
        if bye is None:
            continue
        try:
            out[str(abv).strip().upper()] = int(bye)
        except (TypeError, ValueError):
            continue
    # Alias variants used by some feeds.
    if "WAS" in out and "WSH" not in out:
        out["WSH"] = out["WAS"]
    if "LAR" in out and "LA" not in out:
        out["LA"] = out["LAR"]
    if "JAX" in out and "JAC" not in out:
        out["JAC"] = out["JAX"]
    return out

# --- utils/utils.py L420 ---
def _overlay_players_index(base: Optional[Dict], cache_slot: Dict[str, object]) -> Optional[Dict]:
    """Apply player_current_team DB overlay onto a players index dict."""
    if not isinstance(base, dict):
        return base
    try:
        from data_building.external_data.player_current_team import (
            apply_team_overlay,
            load_current_team_overlay,
        )
    except Exception:
        return base

    overlay = load_current_team_overlay()
    if not overlay:
        return base

    # Version the overlay by size + a few sample values so TTL expiry rebuilds.
    overlay_ver = (len(overlay), sum(hash(f"{k}:{v}") & 0xFFFF for k, v in list(overlay.items())[:32]))
    sig = (id(base), overlay_ver)
    if cache_slot.get("sig") == sig and isinstance(cache_slot.get("data"), dict):
        return cache_slot["data"]  # type: ignore[return-value]

    merged = apply_team_overlay(base, overlay, bye_by_team=_bye_lookup_for_overlay())
    cache_slot["sig"] = sig
    cache_slot["data"] = merged
    return merged

# --- utils/utils.py L448 ---
def load_players_index() -> Optional[Dict]:
    """Returns the cached player index (Sleeper ↔ Tank01/name/team) or None.

    Uses an mtime-guarded in-memory cache: this 1.1 MB file is read dozens of
    times per request, so re-parsing it each time is pure CPU waste. The
    returned dict is shared — callers must treat it as read-only.

    Current NFL team is overlaid from the shared ``player_current_team`` table
    when available, so the web service picks up daily trade/FA updates written
    by cron (cron and web do not share a filesystem).
    """
    base = read_json_cached(path_players_index())
    return _overlay_players_index(base, _PLAYERS_INDEX_MERGED)

# --- utils/utils.py L463 ---
def load_relevant_index() -> Optional[Dict]:
    """Returns the cached relevant-players index or None.

    Uses the same mtime-guarded in-memory cache as load_players_index: the
    player modal fires several endpoints per open and each one was re-reading
    and re-parsing this ~586KB file from disk. The returned dict is shared --
    callers must treat it as read-only.
    """
    base = read_json_cached(path_relevant_index())
    return _overlay_players_index(base, _RELEVANT_INDEX_MERGED)

# --- utils/utils.py L474 ---
def load_usage_table() -> Optional[Dict]:
    # Try today's file first, then fall back to most recent existing file
    today = read_json(path_usage_table())
    if today is not None:
        return today
    candidates = sorted(DATA_DIR.glob("usage_table_*.json"), reverse=True)
    for c in candidates:
        data = read_json(str(c))
        if data is not None:
            return data
    return None

# --- utils/utils.py L487 ---
def load_model_value_table(apply_calibration: bool = True):
    # Return ONLY the parsed JSON data, not the path object
    result = read_json(path_model_value_table())

    # Fallback to database if JSON file doesn't exist.
    # Prefer player_values (current, one-row-per-player) over player_value_history.
    if result is None:
        try:
            from dashboard_services.player_value_history import load_current_values_from_db
            result = load_current_values_from_db()
            if result:
                print(f"[load_model_value_table] Loaded {len(result)} players from player_values table")
        except Exception as e:
            print(f"[load_model_value_table] Failed to load from player_values: {e}")

    if result is None:
        try:
            from dashboard_services.player_value_history import load_latest_value_snapshot
            result = load_latest_value_snapshot()
            if result:
                print(f"[load_model_value_table] Loaded {len(result)} players from player_value_history (fallback)")
        except Exception as e:
            print(f"[load_model_value_table] Failed to load from player_value_history: {e}")

    # Overlay trade-data calibrated values where available.
    # Only applies when called from the web layer (apply_calibration=True).
    # The model-update pipeline passes apply_calibration=False to preserve
    # the raw model prior in player_values.value_1qb.
    if result and apply_calibration:
        try:
            from dashboard_services.player_value_history import load_calibration_overrides
            overrides = load_calibration_overrides()
            if overrides:
                for _p in result:
                    _pid = str(_p.get("id") or "")
                    if _pid in overrides:
                        _cal = overrides[_pid]
                        _p["value"]    = _cal["value"]
                        _p["sf_value"] = _cal["sf_value"]
                        # Overlay size-specific calibrated values when available
                        for _sz in (8, 12, 14):
                            if f"value_{_sz}" in _cal:
                                _p[f"value_{_sz}"]    = _cal[f"value_{_sz}"]
                            if f"sf_value_{_sz}" in _cal:
                                _p[f"sf_value_{_sz}"] = _cal[f"sf_value_{_sz}"]

                # Recompute pos_rank / pos_rank_label after calibration changes values.
                # The JSON ranks are based on raw model values; calibration can reorder
                # players within a position so the labels must be rebuilt.
                from collections import defaultdict as _dd
                _pos_idx: dict = _dd(list)
                for _i, _p in enumerate(result):
                    _pos = str(_p.get("position") or "").upper()
                    if _pos and _pos != "PICK":
                        _pos_idx[_pos].append(_i)
                for _pos, _idxs in _pos_idx.items():
                    _idxs.sort(key=lambda _i: float(result[_i].get("value") or 0), reverse=True)
                    for _rank, _i in enumerate(_idxs, 1):
                        result[_i]["pos_rank"]       = _rank
                        result[_i]["pos_rank_label"] = f"{_pos}{_rank}"
                _sf_pos_idx: dict = _dd(list)
                for _i, _p in enumerate(result):
                    _pos = str(_p.get("position") or "").upper()
                    if _pos and _pos != "PICK":
                        _sf_pos_idx[_pos].append(_i)
                for _pos, _idxs in _sf_pos_idx.items():
                    _idxs.sort(key=lambda _i: float(result[_i].get("sf_value") or 0), reverse=True)
                    for _rank, _i in enumerate(_idxs, 1):
                        result[_i]["sf_pos_rank"]       = _rank
                        result[_i]["sf_pos_rank_label"] = f"{_pos}{_rank}"
        except Exception as _e:
            print(f"[load_model_value_table] Calibration overlay skipped: {_e}")

    return result

# --- utils/utils.py L563 ---
def load_teams_index() -> Optional[Dict]:
    from utils.nfl import canonical_teams_index
    """Returns the cached teams index or None.

    Alias keys such as WSH are merged into WAS so Washington appears once.
    """
    raw = read_json(path_teams_index())
    if not raw:
        return raw
    return canonical_teams_index(raw)

# --- utils/utils.py L574 ---
def load_idp_index() -> Optional[Dict]:
    """Returns the cached teams index or None."""
    return read_json(path_idp_index())

# --- utils/utils.py L579 ---
def load_week_stats(season: int, week: int) -> Optional[Dict]:
    """Returns cached weekly stats or None."""
    return read_json(path_week_stats(season, week))

# --- utils/utils.py L584 ---
def load_week_sched(season: int, week: int) -> Optional[Dict]:
    from utils.nfl import canonicalize_schedule
    """Returns cached weekly schedule or None."""
    return canonicalize_schedule(read_json(path_week_schedule(season, week)))

# --- utils/utils.py L589 ---
    from utils.nfl import canonicalize_schedule
def load_week_schedule(season: int, w: int):
    """
    Load schedule for (season, week), fetching and caching if needed.
    """
    from utils.nfl import canonicalize_schedule
    week_path = Path(path_week_schedule(season, w))
    if not week_path.exists():
        # FIX: use keyword args so order is correct
        get_week_schedule_cached(season=season, week=w, fetch_fn=get_nfl_games_for_week_raw)

    with open(week_path, "r", encoding="utf-8") as f:
        schedule = json.load(f)

    return canonicalize_schedule(schedule)

# --- utils/utils.py L604 ---
def _week_proj_memo_max_from_env() -> int:
    try:
        return max(1, int(os.getenv("WEEK_PROJ_MEMO_MAX", "24")))
    except (TypeError, ValueError):
        return 24

# --- utils/utils.py L611 ---
_WEEK_PROJ_MEMO_MAX = _week_proj_memo_max_from_env()

# --- utils/utils.py L612 ---
_WEEK_PROJ_MEMO: _OrderedDict = _OrderedDict()

# --- utils/utils.py L613 ---
_WEEK_PROJ_MEMO_LOCK = _threading.Lock()

# --- utils/utils.py L616 ---
def _read_week_projection_file(season: int, w: int, use_memo: bool = True) -> Dict:
    """Parse the on-disk cache for (season, week) through the bounded LRU memo.

    Never fetches: returns {} when the file is missing or unreadable.  The
    memo is an LRU keyed by (season, week, mtime); inserting past
    ``_WEEK_PROJ_MEMO_MAX`` evicts only the oldest entries (the previous
    clear-all behavior dropped every week's parsed copy at once and forced
    a re-parse storm on the next requests).
    """
    proj_path = Path(path_week_proj(season, w))
    if not proj_path.exists():
        return {}

    try:
        mtime = proj_path.stat().st_mtime
    except OSError:
        mtime = None
    memo_key = (int(season), int(w), mtime)
    if use_memo:
        with _WEEK_PROJ_MEMO_LOCK:
            cached = _WEEK_PROJ_MEMO.get(memo_key)
            if cached is not None:
                _WEEK_PROJ_MEMO.move_to_end(memo_key)
                return cached

    try:
        with open(proj_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data = data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[projections] load failed for {season} w{w}: {e}")
        return {}
    with _WEEK_PROJ_MEMO_LOCK:
        _WEEK_PROJ_MEMO[memo_key] = data
        _WEEK_PROJ_MEMO.move_to_end(memo_key)
        while len(_WEEK_PROJ_MEMO) > _WEEK_PROJ_MEMO_MAX:
            _WEEK_PROJ_MEMO.popitem(last=False)
    return data

# --- utils/utils.py L656 ---
def load_week_projection(season: int, w: int, force_refresh: bool = False) -> Optional[Dict]:
    from utils.projections import fetch_week_projections
    """
    Load projections cache for (season, week). Returns multi-variant dict or {}.
    """
    proj_path = Path(path_week_proj(season, w))
    if (not proj_path.exists() or force_refresh
            or _week_proj_is_stale(season, w, str(proj_path))):
        try:
            get_week_projections_cached(season, w, fetch_week_projections, force_refresh=force_refresh)
        except Exception as e:
            print(f"[projections] fetch failed for {season} w{w}: {e}")

    return _read_week_projection_file(season, w, use_memo=not force_refresh)

# --- utils/utils.py L671 ---
def save_week_projections(season: int, week: int, proj_map: dict) -> None:
    write_json(path_week_proj(season, week), proj_map)
    # Persist to Redis (fail-soft) so a fresh instance with an empty disk
    # cache can restore this week instead of refetching it from Sleeper.
    try:
        from dashboard_services import proj_store

        proj_store.save(season, week, proj_map)
    except Exception:
        pass

    from utils.nfl import canonicalize_schedule
# --- utils/utils.py L683 ---
def save_week_schedule(season: int, week: int, data: List[Dict]) -> None:
    from utils.nfl import canonicalize_schedule
    write_json(path_week_schedule(season, week), canonicalize_schedule(data))

# --- utils/utils.py L691 ---
_WEEK_PROJ_TTL_HOURS = 1   # re-fetch the live week's projections at most hourly

# --- utils/utils.py L692 ---
_WEEK_PROJ_EMPTY_TTL_SEC = 15 * 60

# --- utils/utils.py L693 ---
_WEEK_PROJ_EMPTY_MAX_BYTES = 16

# --- utils/utils.py L697 ---
_WEEK_PROJ_FAIL_UNTIL: Dict[tuple, float] = {}

# --- utils/utils.py L698 ---
_WEEK_PROJ_FAIL_LOCK = _threading.Lock()

# --- utils/utils.py L701 ---
def _week_proj_file_is_empty(cache_path: str) -> bool:
    """True for a missing file or a failed-fetch ``{}`` placeholder."""
    try:
        return os.path.getsize(cache_path) < _WEEK_PROJ_EMPTY_MAX_BYTES
    except OSError:
        return True

# --- utils/utils.py L709 ---
def _remove_empty_week_proj(cache_path: str) -> None:
    """Delete a ``{}`` placeholder so it cannot look like a real cache hit."""
    if not _week_proj_file_is_empty(cache_path):
        return
    try:
        os.remove(cache_path)
    except OSError:
        pass

# --- utils/utils.py L719 ---
def _week_proj_is_stale(season: int, week: int, cache_path: str) -> bool:
    """True when the cache should be re-fetched.

    Completed weeks (and past seasons) never change, so a *populated* cache
    is permanent. An empty ``{}`` file is never a real projection set — treat
    it as missing so the next request refetches (failed-fetch backoff lives
    in memory, not on disk).
    """
    if _week_proj_file_is_empty(cache_path):
        return True
    try:
        age = time.time() - os.path.getmtime(cache_path)
    except OSError:
        return True
    try:
        state = get_nfl_state() or {}
        cur_season = int(state.get("season") or 0)
        cur_week   = int(state.get("week") or state.get("leg") or 0)
    except Exception:
        return False
    if cur_season == 0 or season < cur_season:
        return False                      # past season → immutable
    if season == cur_season and week < cur_week:
        return False                      # already-completed week → immutable
    return age > _WEEK_PROJ_TTL_HOURS * 3600

# --- utils/utils.py L750 ---
_WEEK_PROJ_BUILD_LOCK_TIMEOUT = 20.0

# --- utils/utils.py L751 ---
_WEEK_PROJ_LOCKS: Dict[tuple, _threading.RLock] = {}

# --- utils/utils.py L752 ---
_WEEK_PROJ_LOCKS_GUARD = _threading.Lock()

# --- utils/utils.py L755 ---
def _week_proj_lock(season: int, week: int) -> _threading.RLock:
    """Per-(season, week) in-process lock from a small guarded registry.

    An RLock because load_week_projection can re-enter
    get_week_projections_cached for the same week on the same thread.
    Distinct (season, week) pairs are bounded in practice (18 weeks per
    season), so the registry needs no eviction.
    """
    key = (int(season), int(week))
    with _WEEK_PROJ_LOCKS_GUARD:
        lock = _WEEK_PROJ_LOCKS.get(key)
        if lock is None:
            lock = _threading.RLock()
            _WEEK_PROJ_LOCKS[key] = lock
        return lock

# --- utils/utils.py L772 ---
@_contextmanager
def _week_proj_cross_lock(season: int, week: int):
    """Yield True when the caller may fetch, False when another process owns it.

    Wraps the cross-process resource_build_lock (Postgres advisory lock, flock
    fallback).  Fail-open by design: if the lock machinery itself errors
    (import failure, no database and no fcntl, ...), the caller proceeds
    under the in-process lock alone rather than losing projections entirely.
    """
    try:
        from dashboard_services.league_singleflight import (
            LeagueBuildBusy,
            resource_build_lock,
        )
    except Exception:
        yield True
        return
    cm = resource_build_lock(
        f"projections:{int(season)}:{int(week)}",
        timeout=_WEEK_PROJ_BUILD_LOCK_TIMEOUT,
    )
    try:
        cm.__enter__()
    except LeagueBuildBusy:
        yield False
        return
    except Exception:
        yield True
        return
    try:
        yield True
    finally:
        try:
            cm.__exit__(None, None, None)
        except Exception:
            pass

# --- utils/utils.py L810 ---
def _restore_week_proj_from_redis(season: int, week: int, cache_path: str) -> Optional[Dict]:
    """Restore a missing week-projection file from Redis, or None.

    Writes the file only (never re-saves to Redis, so restores cannot loop)
    and stamps its mtime with the stored ``saved_at`` so the existing
    staleness logic judges the restored copy exactly like a fetched one.
    Returns the data when the restored file is usable and not stale; None
    sends the caller down the normal Sleeper fetch path.
    """
    try:
        from dashboard_services import proj_store

        hit = proj_store.load(season, week)
    except Exception:
        return None
    if not hit:
        return None
    saved_at, data = hit
    if not isinstance(data, dict) or not data:
        return None
    stamp = saved_at if saved_at and saved_at > 0 else time.time()
    try:
        write_json(cache_path, data)
        os.utime(cache_path, (stamp, stamp))
    except Exception:
        return None
    if _week_proj_is_stale(season, week, cache_path):
        return None
    return _read_week_projection_file(season, week) or data

# --- utils/utils.py L841 ---
def get_week_projections_cached(
        season: int,
        week: int,
        fetch_fn: Callable[[int, int], Dict],
        force_refresh: bool = False,
) -> Dict:
    """
    fetch_fn returns { sleeper_id: {ppr, half_ppr, std, tep, ...} }.

    Concurrency contract: projections are league-independent, so at most one
    thread/process fetches a (season, week) at a time.  Waiters re-check the
    disk cache after acquiring the locks and serve the winner's file.
    ``force_refresh`` is debounced: it refetches only when the file is
    missing, empty, or stale, so per-league live-week refresh calls cannot
    each trigger a Sleeper fetch inside the TTL window.
    """
    cache_path = path_week_proj(season, week)
    memo_key = (int(season), int(week))

    if os.path.exists(cache_path):
        # Serve from disk unless this is the live week and its cache has aged
        # out.  This now applies to force_refresh too (the debounce above).
        # Empty ``{}`` placeholders are always stale and are removed below.
        if not _week_proj_is_stale(season, week, cache_path):
            return load_week_projection(season, week) or {}
        _remove_empty_week_proj(cache_path)

    if not force_refresh:
        with _WEEK_PROJ_FAIL_LOCK:
            fail_until = _WEEK_PROJ_FAIL_UNTIL.get(memo_key, 0.0)
        if fail_until and time.time() < fail_until:
            return {}

    with _week_proj_lock(season, week):
        # Re-check after acquiring: another thread may have just fetched.
        if os.path.exists(cache_path):
            if not _week_proj_is_stale(season, week, cache_path):
                return _read_week_projection_file(season, week)
            _remove_empty_week_proj(cache_path)

        if not force_refresh:
            with _WEEK_PROJ_FAIL_LOCK:
                fail_until = _WEEK_PROJ_FAIL_UNTIL.get(memo_key, 0.0)
            if fail_until and time.time() < fail_until:
                return {}

        with _week_proj_cross_lock(season, week) as may_fetch:
            if not may_fetch:
                # Another process is fetching; serve whatever is on disk
                # (possibly {}) instead of duplicating the fetch.
                return _read_week_projection_file(season, week)

            # Re-check again: the previous lock owner may have written the
            # file while this caller waited on the cross-process lock.
            if os.path.exists(cache_path):
                if not _week_proj_is_stale(season, week, cache_path):
                    return _read_week_projection_file(season, week)
                _remove_empty_week_proj(cache_path)

            if not os.path.exists(cache_path):
                restored = _restore_week_proj_from_redis(season, week, cache_path)
                if restored is not None:
                    return restored

            data = fetch_fn(season, week) or {}
            if data:
                with _WEEK_PROJ_FAIL_LOCK:
                    _WEEK_PROJ_FAIL_UNTIL.pop(memo_key, None)
                save_week_projections(season, week, proj_map=data)
                return data

            # Do not persist ``{}`` — it masquerades as a populated cache
            # after deploys and cron "fresh today" checks. Back off briefly
            # in this process instead.
            with _WEEK_PROJ_FAIL_LOCK:
                _WEEK_PROJ_FAIL_UNTIL[memo_key] = time.time() + _WEEK_PROJ_EMPTY_TTL_SEC
            _remove_empty_week_proj(cache_path)
            return {}

# --- utils/utils.py L921 ---
def get_or_refresh_schedule_path(season: int, week: int) -> Optional[str]:
    path = path_week_schedule(season, week)
    return path if os.path.exists(path) else None

# --- utils/utils.py L926 ---
def get_week_schedule_cached(
        season: int,
        week: int,
        fetch_fn: Callable[[int, int, str], List[Dict]],
        season_type: str = "reg",
) -> List[Dict]:
    """
    fetch_fn should call Tank01 /getNFLSchedule (or your schedule endpoint)
    and return a list[dict] of games.
    """
    cache_path = get_or_refresh_schedule_path(season, week)

    if cache_path is not None:
        # Just load via your schedule loader
        return load_week_schedule(season, week)

    # no cache for today → fetch and save
    data = fetch_fn(week, season, season_type)
    save_week_schedule(season, week, data)
    return data

# --- utils/utils.py L948 ---
def get_players_index_cached(rapidapi_key: str = "") -> Dict[str, Dict[str, Any]]:
    """Load the preserved legacy identity crosswalk without network I/O.

    Sleeper/ESPN metadata owns future refreshes.  The unused argument remains
    for source compatibility with maintenance callers.
    """
    cache_path = CACHE_DIR / "tank01-players_index.json"
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# --- utils/utils.py L1915 ---
def _clear_func_cache_for_league(func: Any, expected_name: str, league_id: str) -> None:
    """
    Remove cache entries for a given league_id from a ttl_cache-decorated function.

    Assumes keys are of the form:
        (func_name, frozen_args, frozen_kwargs)

    and that league_id is passed as the first positional argument.
    """
    if not hasattr(func, "_cache"):
        return

    cache = func._cache
    cache_lock = getattr(func, "_cache_lock", _threading.Lock())
    league_id = str(league_id)

    keys_to_del = []

    with cache_lock:
        cache_keys = list(cache.keys())
    for key in cache_keys:
        # Defensive unpack, in case something else ever gets put in the cache
        try:
            func_name, args, kwargs = key
        except ValueError:
            continue

        if func_name != expected_name:
            continue

        # Our decorator stores frozen_args, so args is a tuple-like of the original args
        if not args:
            continue

        # We call these functions with league_id as the first positional arg
        if str(args[0]) == league_id:
            keys_to_del.append(key)

    with cache_lock:
        for k in keys_to_del:
            cache.pop(k, None)

# --- utils/utils.py L1958 ---
def clear_activity_cache_for_league(league_id: str) -> None:
    """
    Clear only the caches relevant to the Activity page for a given league:
      - transactions
      - traded picks
      - users/rosters
    """

    # 1) Clear transactions for all weeks in this league
    _clear_func_cache_for_league(get_transactions, "get_transactions", league_id)

    # 2) Clear traded picks for this league
    _clear_func_cache_for_league(get_traded_picks, "get_traded_picks", league_id)

    # 3) Clear users/rosters for this league (more aggressive: wipe entire cache)
    _clear_func_cache_for_league(get_users, "get_users", league_id)

    _clear_func_cache_for_league(get_rosters, "get_rosters", league_id)

# --- utils/utils.py L1978 ---
def clear_league_provider_cache_for_league(league_id: str) -> None:
    """Evict only cached Sleeper payloads that feed a league-context rebuild."""
    for func, name in (
        (_fetch_league, "_fetch_league"),
        (get_users, "get_users"),
        (get_rosters, "get_rosters"),
        (get_matchups, "get_matchups"),
        (get_transactions, "get_transactions"),
        (get_traded_picks, "get_traded_picks"),
    ):
        _clear_func_cache_for_league(func, name, league_id)

# --- utils/utils.py L1991 ---
def clear_teams_cache_for_league(league_id: str) -> None:
    try:
        _clear_func_cache_for_league(get_users, "get_users", league_id)
        _clear_func_cache_for_league(get_rosters, "get_rosters", league_id)
    except Exception as e:
        print("clear_weekly_cache_for_league RAISED EXCEPTION:", repr(e))
        traceback.print_exc()

# --- utils/utils.py L2000 ---
def clear_weekly_cache_for_league(league_id: str) -> None:
    # Weekly page touches NFL state, players, users, rosters.
    # If you later make them per-league, you can reuse _clear_func_cache_for_league.
    try:
        # get_nfl_state / get_nfl_players take no league arg, so the per-league
        # clearer can never match their (empty-args) cache key — it's a no-op.
        # Use the decorator's clear_cache() to actually evict the global entry.
        get_nfl_state.clear_cache()
        get_nfl_players.clear_cache()
        _clear_func_cache_for_league(get_users, "get_users", league_id)
        _clear_func_cache_for_league(get_rosters, "get_rosters", league_id)
        _clear_func_cache_for_league(get_matchups, "get_matchups", league_id)
    except Exception as e:
        print("clear_weekly_cache_for_league RAISED EXCEPTION:", repr(e))
        traceback.print_exc()
