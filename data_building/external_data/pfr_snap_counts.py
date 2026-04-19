"""
ESPN snap count integration module.

Fetches offensive snap counts from ESPN's player gamelog API.
Falls back to usage-based estimation when the API is unavailable.

Endpoint:
  https://site.api.espn.com/apis/site/v2/sports/football/nfl/athletes/{id}/gamelog
  ?season={year}

Data is cached to CACHE_DIR/espn_snap_counts_{season}.json so the season-long
scrape (one request per skill player) only happens once per day.
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional

import requests

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "snap_counts"

_ESPN_GAMELOG = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
    "/athletes/{espn_id}/gamelog"
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.espn.com/",
}

# Skill positions we care about
_SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}

# Position-specific coefficients for snap share estimation (fallback only)
SNAP_SHARE_COEFFICIENTS = {
    "QB": 0.95,
    "RB": 0.55,
    "WR": 0.70,
    "TE": 0.65,
}
AVG_TOUCHES_FEATURED = {
    "QB": 35,
    "RB": 18,
    "WR": 8,
    "TE": 6,
}


def estimate_snap_share_from_usage(
        position: str,
        avg_targets: float,
        avg_carries: float,
        avg_pass_att: float = 0.0,
) -> float:
    """
    Estimate offensive snap share from usage stats when real data is unavailable.

    Returns estimated snap share (0-1).
    """
    if position not in SNAP_SHARE_COEFFICIENTS:
        return 0.0

    touches = avg_pass_att if position == "QB" else avg_targets + avg_carries
    if touches == 0:
        return 0.0

    avg_featured = AVG_TOUCHES_FEATURED.get(position, 10)
    touch_ratio = min(touches / avg_featured, 1.5)
    estimated = touch_ratio * SNAP_SHARE_COEFFICIENTS[position]
    return min(max(estimated, 0.0), 1.0)


# ---------------------------------------------------------------------------
# ESPN gamelog helpers
# ---------------------------------------------------------------------------

def _fetch_espn_gamelog_snaps(
        espn_id: str,
        season: int,
        week_set: set,
        session: requests.Session,
) -> Optional[Dict]:
    """
    Call ESPN's player gamelog endpoint for one player and extract snap counts.

    Returns {avg_snaps, total_snaps, avg_snap_pct, games_played} or None.

    ESPN gamelog structure (regular season = seasonType id "2"):
      seasonTypes[].categories[].labels   → ["DATE", "OPP", ..., "SNP", "SNPP"]
      seasonTypes[].categories[].events[] → [{id, stats: [values...]}]
      events{}                            → {eventId: {week, ...}}
    """
    url = _ESPN_GAMELOG.format(espn_id=espn_id)
    try:
        resp = session.get(url, params={"season": season}, timeout=10)
        if not resp.ok:
            return None
        data = resp.json()
    except Exception:
        return None

    # Build week lookup: event_id → week number
    event_week: Dict[str, int] = {}
    for evt_id, evt in (data.get("events") or {}).items():
        event_week[str(evt_id)] = int(evt.get("week") or 0)

    # Regular season = seasonType id "2"
    season_type = next(
        (st for st in (data.get("seasonTypes") or []) if str(st.get("id")) == "2"),
        None,
    )
    if not season_type:
        return None

    for category in season_type.get("categories") or []:
        raw_labels = category.get("labels") or []
        labels_upper = [str(l).upper() for l in raw_labels]

        # SNP = snaps played, SNPP = snap percentage
        snp_idx = next(
            (i for i, l in enumerate(labels_upper) if l in ("SNP", "SNPS", "SNAPS")),
            None,
        )
        snpp_idx = next(
            (i for i, l in enumerate(labels_upper) if "SNP" in l and "%" in l),
            None,
        )
        # some ESPN responses use "SNPP" without a literal "%" character
        if snpp_idx is None:
            snpp_idx = next(
                (i for i, l in enumerate(labels_upper) if l in ("SNPP", "SNAP%", "SNP%")),
                None,
            )

        if snp_idx is None:
            continue

        snaps_list: list[int] = []
        pct_list: list[float] = []

        for event_entry in category.get("events") or []:
            evt_id = str(event_entry.get("id") or "")
            week = event_week.get(evt_id, 0)
            if week_set and week not in week_set:
                continue

            stats = event_entry.get("stats") or []
            try:
                snaps = int(stats[snp_idx])
            except (ValueError, TypeError, IndexError):
                continue

            if snaps <= 0:
                continue

            snaps_list.append(snaps)

            if snpp_idx is not None and snpp_idx < len(stats):
                try:
                    pct_raw = float(str(stats[snpp_idx]).rstrip("%"))
                    # ESPN returns 0-100 or 0-1 depending on the endpoint version
                    pct = pct_raw / 100.0 if pct_raw > 1.0 else pct_raw
                    pct_list.append(pct)
                except (ValueError, TypeError):
                    pass

        if not snaps_list:
            continue

        return {
            "avg_snaps": sum(snaps_list) / len(snaps_list),
            "total_snaps": sum(snaps_list),
            "avg_snap_pct": sum(pct_list) / len(pct_list) if pct_list else 0.0,
            "games_played": len(snaps_list),
        }

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_season_snap_counts(
        season: int,
        weeks: range = range(1, 19),
        players_index: Optional[Dict] = None,
) -> Dict[str, Dict]:
    """
    Fetch offensive snap counts for all skill-position players via ESPN gamelog.

    Results are cached to CACHE_DIR/espn_snap_counts_{season}.json and reused
    for the rest of the day so repeated calls within a single build don't
    re-hit ESPN's API.

    Args:
        season:         NFL season year (e.g. 2024).
        weeks:          Regular-season weeks to include (default 1-18).
        players_index:  Optional pre-loaded players_index dict.  If None the
                        function loads it from cache automatically.

    Returns:
        Dict keyed by player name:
            {avg_off_snap_pct, avg_off_snaps, total_off_snaps, position, team, games_played}
    """
    cache_file = CACHE_DIR / f"espn_snap_counts_{season}.json"

    # Return today's cached file if it exists
    if cache_file.exists():
        try:
            mtime_date = datetime.fromtimestamp(cache_file.stat().st_mtime).date()
            if mtime_date == date.today():
                with cache_file.open("r", encoding="utf-8") as fh:
                    cached = json.load(fh)
                print(f"[snap_counts] Loaded {len(cached)} players from cache ({season})")
                return cached
        except Exception:
            pass  # stale or corrupt cache — re-fetch

    # Load players_index if not supplied
    if players_index is None:
        try:
            from utils.utils import load_players_index
            players_index = load_players_index() or {}
        except Exception as exc:
            print(f"[snap_counts] Could not load players_index: {exc}")
            return {}

    week_set = set(weeks)
    session = requests.Session()
    session.headers.update(_HEADERS)

    result: Dict[str, Dict] = {}
    total_queried = 0

    for pid, player in players_index.items():
        espn_id = player.get("espnID") or player.get("espn_id")
        if not espn_id:
            continue

        position = player.get("pos", "")
        if position not in _SKILL_POSITIONS:
            continue

        name = player.get("name", "")
        team = player.get("team", "") or ""

        snap_info = _fetch_espn_gamelog_snaps(str(espn_id), season, week_set, session)

        if snap_info and snap_info["games_played"] > 0:
            result[name] = {
                "avg_off_snap_pct": snap_info["avg_snap_pct"],
                "avg_off_snaps": snap_info["avg_snaps"],
                "total_off_snaps": snap_info["total_snaps"],
                "position": position,
                "team": team,
                "games_played": snap_info["games_played"],
            }

        total_queried += 1
        if total_queried % 100 == 0:
            print(f"[snap_counts] {total_queried} players queried, {len(result)} with snap data…")

        time.sleep(0.12)  # ~8 req/s — polite to ESPN's unofficial API

    print(f"[snap_counts] ESPN: {len(result)}/{total_queried} players have snap data ({season})")

    # Persist cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with cache_file.open("w", encoding="utf-8") as fh:
        json.dump(result, fh)

    return result


if __name__ == "__main__":
    snap_data = fetch_season_snap_counts(2024, weeks=range(1, 19))

    top = sorted(snap_data.items(), key=lambda x: x[1]["total_off_snaps"], reverse=True)
    print(f"\nTop 10 players by total snaps ({len(snap_data)} total):")
    for player_name, d in top[:10]:
        print(
            f"  {player_name} ({d['position']}, {d['team']}): "
            f"{d['total_off_snaps']} snaps, "
            f"{d['avg_off_snap_pct']:.1%} avg snap%"
        )
