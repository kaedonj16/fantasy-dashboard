from __future__ import annotations

import html
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, date
from typing import Dict, Any, Iterable, Tuple, Optional, List, Union

import numpy as np
import pandas as pd
import requests

from dashboard_services.api import (
    avatar_url,
    get_nfl_state,
    avatar_from_users,
    team_avatar,
)
from dashboard_services.matchups import build_matchup_preview
from dashboard_services.platform_api import get_matchups, get_transactions as platform_get_transactions
from dashboard_services.players import build_roster_display_maps
from dashboard_services.team_crest import team_crest_data_uri
from utils.utils import safe_owner_name

_NFL_CITY: dict[str, str] = {
    "ARI": "Arizona", "ATL": "Atlanta", "BAL": "Baltimore", "BUF": "Buffalo",
    "CAR": "Carolina", "CHI": "Chicago", "CIN": "Cincinnati", "CLE": "Cleveland",
    "DAL": "Dallas", "DEN": "Denver", "DET": "Detroit", "GB": "Green Bay",
    "HOU": "Houston", "IND": "Indianapolis", "JAX": "Jacksonville",
    "KC": "Kansas City", "LAC": "LA Chargers", "LAR": "LA Rams",
    "LV": "Las Vegas", "MIA": "Miami", "MIN": "Minnesota",
    "NE": "New England", "NO": "New Orleans", "NYG": "NY Giants", "NYJ": "NY Jets",
    "PHI": "Philadelphia", "PIT": "Pittsburgh", "SEA": "Seattle",
    "SF": "San Francisco", "TB": "Tampa Bay", "TEN": "Tennessee",
    "WAS": "Washington", "WSH": "Washington",
}

def _team_city(abbr: str) -> str:
    return _NFL_CITY.get((abbr or "").upper(), abbr)


def matchup_cards_last_week(
        league_id: str,
        df_weekly: pd.DataFrame,
        roster_map: dict,
        players_map: dict,
        rosters: list,
        users: list,
        platform: str,
        season: str,
        roster_positions: list = None,
) -> tuple[int, str, dict]:
    """
    Returns: (week_number, html_for_matchup_cards, top_by_pos_dict)
      top_by_pos_dict: {'QB': [ {name, pts, nfl, team, owner}, ... up to 3 ], ...}
    """
    last_week = int(df_weekly["week"].max())
    raw = get_matchups(platform, league_id, last_week, season) or []

    # group rows per matchup_id
    by_mid: dict[Any, list] = defaultdict(list)
    for r in raw:
        by_mid[r.get("matchup_id")].append(r)

    # precompute quick lookups
    roster_by_id = {str(r.get("roster_id")): r for r in rosters}
    user_by_id = {u["user_id"]: u for u in users}
    display_name_by_owner = {uid: u.get("display_name") for uid, u in user_by_id.items()}

    # record + avatar lookups by roster_id string
    record_by_rid: dict[str, tuple[int, int]] = {}
    avatar_by_rid: dict[str, Optional[str]] = {}
    for r in rosters:
        rid = str(r.get("roster_id"))
        settings = r.get("settings") or {}
        record_by_rid[rid] = (
            settings.get("wins") or 0,
            settings.get("losses") or 0,
        )
        # Team picture (roster- or user-level) -> crest -> profile picture.
        avatar_by_rid[rid] = team_avatar(platform, r, users)

    buckets: dict[str, list] = defaultdict(list)

    def pmeta(pid: str):
        p = players_map.get(str(pid), {})
        getp = p.get
        name = getp("name") or str(pid)
        nfl = getp("team") or "FA"
        pos = getp("pos") or (getp("fantasy_positions") or [""])[0]
        if pid.isalpha() and 2 <= len(pid) <= 3 and not pos:
            pos, name, nfl = "DEF", f"{pid} D/ST", pid
        return name, nfl, pos

    cards = []
    for mid, rows in by_mid.items():
        if not rows or mid is None or mid == 0:
            continue
        rows = sorted(rows, key=lambda r: str(r.get("roster_id")))
        L = rows[0]
        R = rows[1] if len(rows) > 1 else {}

        ridL = str(L.get("roster_id"))
        ridR = str(R.get("roster_id"))

        ownerL = roster_by_id.get(ridL, {}).get("owner_id")
        ownerR = roster_by_id.get(ridR, {}).get("owner_id")

        username = display_name_by_owner.get(ownerL)
        username2 = display_name_by_owner.get(ownerR)

        ln = safe_owner_name(roster_map, L.get("roster_id"))
        rn = safe_owner_name(roster_map, R.get("roster_id"))
        lp = float(L.get("points") or 0.0)
        rp = float(R.get("points") or 0.0)

        avatar = avatar_by_rid.get(ridL)
        avatar2 = avatar_by_rid.get(ridR)

        winsL, lossesL = record_by_rid.get(ridL, (0, 0))
        winsR, lossesR = record_by_rid.get(ridR, (0, 0))

        def harvest(row, owner_name: str):
            starters = [s for s in (row.get("starters") or []) if s]
            spts = row.get("starters_points") or []
            players_points = row.get("players_points") or {}
            for i, pid in enumerate(starters):
                pid_s = str(pid)
                if i < len(spts) and spts[i] is not None:
                    pts = float(spts[i])
                else:
                    pts = float(players_points.get(pid_s, 0.0))
                name, nfl, pos = pmeta(pid_s)
                if pos:
                    buckets[pos].append(
                        {
                            "name": name,
                            "pts": pts,
                            "nfl": nfl,
                            "owner": owner_name,
                            "pid": pid_s,
                        }
                    )

        harvest(L, ln)
        if R:
            harvest(R, rn)

        l_cls = "win" if lp > rp else "loss" if rp > lp else "tie"
        r_cls = "win" if rp > lp else "loss" if lp > rp else "tie"

        cards.append(
            f"""
        <div class="mu-card">
          <div class="mu-row">
            <div class="mu-team {l_cls}">
              <div style="display: flex; align-items: center; gap: 5px;">
                <img class="avatar" src="{avatar}" alt="" loading="lazy" decoding="async" onerror="this.style.display='none'">
                <div class="mu-name left"><div style="display: flex; justify-content: flex-start;">{ln}</div><div style="font-weight: 400; font-size: small;">{winsL}-{lossesL} • @{username}</div></div>
              </div>
              <div class="mu-score">{lp:.2f}</div>
            </div>
            <div class="mu-vs">vs</div>
            <div class="mu-team {r_cls}">
              <div class="mu-score">{rp:.2f}</div>
              <div style="display: flex; align-items: center; justify-content: flex-end; gap: 5px;">
                <div class="mu-name right"><div style="display: flex; justify-content: flex-end;">{rn}</div><div style="font-weight: 400; font-size: small">@{username2} • {winsR}-{lossesR}</div></div>
                <img class="avatar" src="{avatar2}" alt="" loading="lazy" decoding="async" onerror="this.style.display='none'">
              </div>
            </div>
          </div>
        </div>
        """
        )

    _base_positions = ["QB", "RB", "WR", "TE", "K", "DEF"]
    rp_set = set(roster_positions or [])
    want_positions = [
        p for p in _base_positions
        if p not in ("K", "DEF") or p in rp_set
    ]
    top_by_pos = {}
    for pos in want_positions:
        pool = sorted(buckets.get(pos, []), key=lambda x: x["pts"], reverse=True)[:3]
        top_by_pos[pos] = pool

    return last_week, "".join(cards), top_by_pos


def fantasy_team_and_roster_for_player(pid: str, rosters: list, roster_map: dict) -> tuple[str, str]:
    """
    Returns (team_name, roster_id) for a player.
    Returns ("Free Agent", "") if player is not rostered.
    """
    for r in rosters:
        if pid in (r.get("players") or []):
            rid = str(r["roster_id"])
            team_name = roster_map.get(rid, f"Roster {rid}")
            return team_name, rid
    return "Free Agent", ""


def render_top_three(top_by_pos: dict, rosters, roster_map, positions: list = None) -> str:
    def card(pos, rows):
        if not rows:
            return f"<div class='side-card'><h2>{pos}</h2><div class='muted'>No data</div></div>"
        lis = []
        for i, r in enumerate(rows, start=1):
            team = r.get("nfl") or r.get("team", "")
            pts = r.get("pts") or r.get("points", 0.0)
            if not r.get("owner_id"):
                owner, roster_id = fantasy_team_and_roster_for_player(r["pid"], rosters, roster_map)
            else:
                owner = "Unknown"
                roster_id = ""
            place = "first" if i == 1 else "second" if i == 2 else "third"

            # Make player name clickable
            pid = r.get("pid", "")
            player_name = r['name']
            clickable_attrs = f" class='name player-clickable' style='cursor:pointer;' data-player-id='{pid}' data-player-name='{player_name}'" if pid else " class='name'"

            # Make team name clickable
            team_clickable = f"<span class='team-clickable' style='cursor:pointer;' data-roster-id='{roster_id}' data-team-name='{owner}'>{owner}</span>" if roster_id else owner

            lis.append(
                f"<div class='side-row'>"
                f"  <span class='rank rank-{place}'>{i}</span>"
                f"  <div class='who'>"
                f"    <div{clickable_attrs}>{player_name}</div>"
                f"    <div class='sub'>{team} • {team_clickable}</div>"
                f"  </div>"
                f"  <div class='pts'>{pts:.1f}</div>"
                f"</div>"
            )
        return f"<div class='side-card'><h3>{pos}</h3>{''.join(lis)}</div>"

    _render_positions = positions if positions else ["QB", "RB", "WR", "TE", "K", "DEF"]
    blocks = [card(pos, top_by_pos.get(pos, [])) for pos in _render_positions]
    return "<div class='sidebar-grid'>" + "".join(blocks) + "</div>"


def _seed_zero_standings(owner_avatar: dict) -> pd.DataFrame:
    """0-0 / 0 PF rows for every known team when no weeks are finalized yet.

    Week 1 (and any stretch before the first finalized leg) used to leave
    ``team_stats`` empty, which painted Standings as "No standings data
    available" even though rosters and Season Hub were live. Seeding keeps the
    table present; Value & Production Share already treats PF==0 as projected.
    """
    owners = [o for o in (owner_avatar or {}) if o]
    if not owners:
        return pd.DataFrame()
    rows = []
    for owner in owners:
        rows.append({
            "owner": owner,
            "Wins": 0,
            "Losses": 0,
            "Ties": 0,
            "G": 0,
            "Win%": 0.0,
            "PF": 0.0,
            "PA": 0.0,
            "AVG": 0.0,
            "MAX": 0.0,
            "MIN": 0.0,
            "STD": 0.0,
            "Record": "0-0",
            "Last3": 0.0,
            "Z_WinPercentage": 0.0,
            "Z_Avg": 0.0,
            "Z_Last3": 0.0,
            "Z_Consistency": 0.0,
            "Z_Ceiling": 0.0,
            "PowerScore": 0.0,
            "StreakType": "",
            "StreakLen": 0,
            "Streak": "",
            "avatar": (owner_avatar or {}).get(owner),
        })
    return pd.DataFrame(rows)


def regular_season_length(settings: dict | None = None) -> int:
    """Regular-season week count from league settings (playoff_week_start - 1).

    Falls back to the request-scoped league settings, then to 14 (Sleeper's
    default 15-week playoff start). Playoff weeks must not leak into SOS.
    """
    src = settings if isinstance(settings, dict) else None
    if not src:
        try:
            from dashboard_services.api import get_league_settings
            src = get_league_settings() or {}
        except Exception:
            src = {}
    try:
        pws = int((src or {}).get("playoff_week_start") or 15)
    except (TypeError, ValueError):
        pws = 15
    if pws <= 1:
        return 14
    return pws - 1


def finalize_team_stats(
        df_finalized: pd.DataFrame,
        owner_avatar: dict,
        matchups_by_week: dict,
        users: list[dict],
        last_week: int,
        regular_season_weeks: int | None = None,
) -> pd.DataFrame:
    """Build the full standings/power team_stats table from finalized weekly
    rows (records, PF/PA/AVG, performance PowerScore, strength of schedule, and
    current streaks).

    Shared by the season builder (build_tables) and the standings week-selector,
    so a "through week N" view is produced by simply passing a week-capped
    ``df_finalized`` and ``last_week=N`` here — both paths agree by construction.
    """
    if df_finalized is None or getattr(df_finalized, "empty", True):
        seeded = _seed_zero_standings(owner_avatar)
        if not seeded.empty:
            return seeded

    records = _compute_team_records(df_finalized.copy())
    team_stats = _aggregate_team_stats(df_finalized.copy(), records)

    if team_stats.empty:
        seeded = _seed_zero_standings(owner_avatar)
        if not seeded.empty:
            return seeded

    team_stats = team_stats.merge(
        pd.Series(owner_avatar, name="avatar", dtype="object"),
        left_on="owner",
        right_index=True,
        how="left",
    )

    last3 = (
        df_finalized.sort_values(["owner", "week"])
        .groupby("owner")["points"]
        .apply(lambda s: s.tail(3).mean() if len(s) else 0.0)
        .rename("Last3")
        .reset_index()
    )
    team_stats = team_stats.merge(last3, on="owner", how="left")
    team_stats["Last3"] = team_stats["Last3"].fillna(0.0)

    def _z(series):
        s = pd.Series(series, dtype="float64")
        sd = float(s.std(ddof=0))
        if sd == 0 or np.isnan(sd):
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / sd

    if "Win%" in team_stats.columns:
        win_pct = team_stats["Win%"].fillna(0.0)
    else:
        if "Ties" in team_stats.columns:
            ties = team_stats["Ties"].fillna(0.0)
        else:
            ties = 0.0
        win_pct = ((team_stats["Wins"] + 0.5 * ties) / team_stats["G"].replace(0, np.nan)).fillna(0.0)

    avg_pts = team_stats.get("AVG", pd.Series(0.0, index=team_stats.index)).fillna(0.0)
    _std_col = team_stats["STD"] if "STD" in team_stats.columns else pd.Series(0.0, index=team_stats.index)
    cons_inv = -_std_col.fillna(_std_col.mean() if len(_std_col) else 0.0)
    ceiling = team_stats.get("MAX", pd.Series(0.0, index=team_stats.index)).fillna(0.0)
    last3_series = team_stats["Last3"].fillna(0.0)

    team_stats["Z_WinPercentage"] = _z(win_pct)
    team_stats["Z_Avg"] = _z(avg_pts)
    team_stats["Z_Last3"] = _z(last3_series)
    team_stats["Z_Consistency"] = _z(cons_inv)
    team_stats["Z_Ceiling"] = _z(ceiling)

    # Weights: recency (Last3) promoted; raw Avg and Win% trimmed slightly
    # so a hot team closing strong is rewarded over a stale season-long average.
    W_WIN, W_AVG, W_LAST3, W_CONS, W_CEIL = 0.15, 0.25, 0.25, 0.15, 0.20
    team_stats["Win%"] = win_pct
    team_stats["PowerScore"] = (
            W_WIN * team_stats["Z_WinPercentage"]
            + W_AVG * team_stats["Z_Avg"]
            + W_LAST3 * team_stats["Z_Last3"]
            + W_CONS * team_stats["Z_Consistency"]
            + W_CEIL * team_stats["Z_Ceiling"]
    )

    sos = build_team_strength(team_stats)
    _reg_weeks = (
        regular_season_length()
        if regular_season_weeks is None
        else int(regular_season_weeks)
    )
    sos_dict = compute_sos_by_team(
        matchups_by_week,
        sos,
        last_week,
        users,
        regular_season_weeks=_reg_weeks,
        past_pairs=owner_pairs_from_weekly(df_finalized),
    )
    sos_df = (
        pd.DataFrame.from_dict(sos_dict, orient="index")
        .reset_index()
        .rename(columns={"index": "owner"})
    )
    team_stats = team_stats.merge(sos_df, on="owner", how="left")
    for _col in ("past_sos", "ros_sos", "past_cnt", "ros_cnt"):
        if _col in team_stats.columns:
            team_stats[_col] = team_stats[_col].fillna(0.0)

    streaks_df = compute_streaks(df_finalized.copy())
    team_stats = team_stats.merge(streaks_df, on="owner", how="left")

    team_stats["StreakType"] = team_stats["StreakType"].fillna("")
    team_stats["StreakLen"] = team_stats["StreakLen"].fillna(0).astype(int)
    team_stats["Streak"] = team_stats["Streak"].fillna("")

    return team_stats


def build_tables(
        league_id: str,
        max_week: int,
        players: dict,
        users: list[dict],
        rosters: list[dict],
        season,
        platform
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Fetch and process league data into DataFrames."""

    user_by_id = {u["user_id"]: u for u in users}

    user_fallback = {
        u["user_id"]: (
                (u.get("metadata") or {}).get("team_name")
                or u.get("display_name")
                or u.get("username")
                or str(u["user_id"])
        )
        for u in users
    }

    roster_map: dict[str, str] = {}
    for r in rosters:
        rid = str(r["roster_id"])
        owner_id = r.get("owner_id")
        roster_map[rid] = (r.get("metadata") or {}).get("team_name") or user_fallback.get(
            owner_id, f"Roster {rid}"
        )

    matchups_by_week = build_matchups_by_week(league_id, range(1, 18), roster_map, players, season, platform)

    # precompute owner_avatar using user_by_id only (no extra scan over users)
    owner_avatar: dict[str, Union[str, None]] = {}
    for r in rosters:
        rid = str(r["roster_id"])
        owner_id = r.get("owner_id")
        display = roster_map.get(rid, f"Roster {rid}")

        # Team picture (roster- or user-level) -> crest -> profile picture.
        owner_avatar[display] = team_avatar(platform, r, users)

    def _fetch_week(week: int) -> list[dict]:
        try:
            week_data = get_matchups(platform, league_id, week, season) or []
        except Exception:
            return []
        rows = []
        for m in week_data:
            rid = str(m.get("roster_id"))
            rows.append({
                "week": week,
                "matchup_id": m.get("matchup_id"),
                "roster_id": rid,
                "owner": roster_map.get(rid, f"Roster {rid}"),
                "points": float(m.get("points", 0.0)),
            })
        return rows

    weeks_to_fetch = list(range(1, max_week + 1))
    weekly_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(len(weeks_to_fetch), 8)) as pool:
        futures = {pool.submit(_fetch_week, w): w for w in weeks_to_fetch}
        for fut in as_completed(futures):
            weekly_rows.extend(fut.result())

    df_weekly = pd.DataFrame(weekly_rows)
    if df_weekly.empty:
        print("Warning: No matchup data found.")
        # Still seed 0-0 standings from rosters so Week 1 ESPN/Yahoo leagues
        # with a delayed schedule feed aren't a blank Standings page.
        seeded = _seed_zero_standings(owner_avatar)
        return pd.DataFrame(), seeded, roster_map
        # raise SystemExit("No matchup data found. Check league ID and weeks.")

    df_weekly["points_against"] = np.nan
    for (_, _mid), grp in df_weekly.groupby(["week", "matchup_id"]):
        if len(grp) == 2:
            i1, i2 = grp.index.tolist()
            p1, p2 = df_weekly.loc[i1, "points"], df_weekly.loc[i2, "points"]
            df_weekly.loc[i1, "points_against"] = p2
            df_weekly.loc[i2, "points_against"] = p1

    if "owner" in df_weekly.columns:
        df_weekly["avatar"] = df_weekly["owner"].map(owner_avatar)
    else:
        df_weekly["avatar"] = None

    _state = get_nfl_state() or {}
    state_season_type = (_state.get("season_type") or "").lower()
    live_season = int(_state.get("season") or datetime.now().year)
    current_leg = int(_state.get("leg") or _state.get("week") or 0)

    if season < live_season:
        df_weekly["finalized"] = True
    elif season == live_season and state_season_type == "off":
        df_weekly["finalized"] = True
    else:
        df_weekly["finalized"] = df_weekly["week"] < current_leg

    finalized_mask = df_weekly["finalized"] == True
    df_finalized = df_weekly[finalized_mask].copy()

    # SOS Past is "games already played": use the last *finalized* week, not
    # the max week in df_weekly (which includes the in-progress current week).
    if not df_finalized.empty and "week" in df_finalized.columns:
        last_week = int(df_finalized["week"].max())
    else:
        last_week = 0
    team_stats = finalize_team_stats(
        df_finalized, owner_avatar, matchups_by_week, users, last_week
    )

    return df_weekly, team_stats, roster_map


def _compute_team_records(df: pd.DataFrame) -> pd.DataFrame:
    wins = defaultdict(int)
    losses = defaultdict(int)
    ties = defaultdict(int)
    games_played = defaultdict(int)

    for (_, _mid), group in df.groupby(["week", "matchup_id"]):
        matchups = list(group.to_dict("records"))
        if len(matchups) != 2:
            continue
        team1, team2 = matchups
        owner1, owner2 = team1["owner"], team2["owner"]
        points1 = float(team1.get("points", 0.0))
        points2 = float(team2.get("points", 0.0))

        games_played[owner1] += 1
        games_played[owner2] += 1

        if points1 > points2:
            wins[owner1] += 1
            losses[owner2] += 1
        elif points2 > points1:
            wins[owner2] += 1
            losses[owner1] += 1
        else:
            ties[owner1] += 1
            ties[owner2] += 1

    results = []
    owners = sorted(set(df["owner"])) if "owner" in df.columns else []
    for owner in owners:
        w = wins[owner]
        l = losses[owner]
        t = ties[owner]
        g = games_played[owner]
        results.append(
            {
                "owner": owner,
                "Wins": w,
                "Losses": l,
                "Ties": t,
                "G": g,
                "Win%": (w + 0.5 * t) / g if g else 0.0,
            }
        )
    # Always return the expected schema so downstream merges on "owner" work even
    # in the preseason when there are no games played yet (empty results).
    return pd.DataFrame(results, columns=["owner", "Wins", "Losses", "Ties", "G", "Win%"])


def _aggregate_team_stats(df_weekly: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    stats = (
        df_weekly.groupby("owner")
        .agg(
            PF=("points", "sum"),
            PA=("points_against", "sum"),
            AVG=("points", "mean"),
            MAX=("points", "max"),
            MIN=("points", "min"),
            STD=("points", "std"),
        )
        .reset_index()
    )

    team_stats = stats.merge(records, on="owner", how="left")

    # Owners with no games yet (preseason) come through the left-merge as NaN.
    for _col in ("Wins", "Losses", "Ties", "G"):
        if _col in team_stats.columns:
            team_stats[_col] = team_stats[_col].fillna(0)

    team_stats["Record"] = team_stats[["Wins", "Losses", "Ties"]].apply(
        lambda r: f"{int(r.Wins)}-{int(r.Losses)}"
                  + (f"-{int(r.Ties)}" if r.Ties else ""),
        axis=1,
    ) if not team_stats.empty else pd.Series(dtype=str)

    return team_stats


def build_matchups_by_week(league_id, weeks, roster_map, players_map, season, platform):
    week_list = list(weeks)

    def _fetch(w: int) -> tuple[int, list]:
        try:
            return w, build_matchup_preview(
                league_id=league_id,
                week=w,
                roster_map=roster_map,
                players_map=players_map,
                season=season,
                platform=platform,
            ) or []
        except Exception:
            return w, []

    by_week: dict[int, list] = {}
    with ThreadPoolExecutor(max_workers=min(len(week_list), 8)) as pool:
        for w, matchups in pool.map(_fetch, week_list):
            by_week[w] = matchups
    return by_week


def _weekly_results_from_df(df_weekly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (_, mid), g in df_weekly.groupby(["week", "matchup_id"]):
        g = g.sort_values("roster_id")
        if len(g) != 2:
            continue
        a, b = g.iloc[0], g.iloc[1]
        pa, pb = float(a.get("points", 0.0)), float(b.get("points", 0.0))
        if pa > pb:
            rows.append({"owner": a["owner"], "week": int(a["week"]), "result": "W"})
            rows.append({"owner": b["owner"], "week": int(b["week"]), "result": "L"})
        elif pb > pa:
            rows.append({"owner": b["owner"], "week": int(b["week"]), "result": "W"})
            rows.append({"owner": a["owner"], "week": int(a["week"]), "result": "L"})
        else:
            rows.append({"owner": a["owner"], "week": int(a["week"]), "result": "T"})
            rows.append({"owner": b["owner"], "week": int(b["week"]), "result": "T"})
    return pd.DataFrame(rows)


def _current_streak(series_results: list[str]) -> tuple[str, int]:
    if not series_results:
        return ("", 0)
    last = series_results[-1]
    n = 1
    for r in reversed(series_results[:-1]):
        if r == last:
            n += 1
        else:
            break
    return (last, n)


def compute_streaks(df_weekly: pd.DataFrame) -> pd.DataFrame:
    res = _weekly_results_from_df(df_weekly)
    if res.empty:
        return pd.DataFrame(columns=["owner", "StreakType", "StreakLen", "Streak"])

    out = []
    for owner, g in res.sort_values("week").groupby("owner"):
        typ, length = _current_streak(g["result"].tolist())
        label = f"{typ}{length}" if typ and length else ""
        out.append(
            {"owner": owner, "StreakType": typ, "StreakLen": int(length), "Streak": label}
        )
    return pd.DataFrame(out)


def get_transactions_by_week(
        league_id: str,
        season_weeks: list[int],
        platform: str = "sleeper",
        season: int = 0,
) -> dict[int, list[dict]]:
    results: dict[int, list[dict]] = {}

    def _fetch(w: int):
        tx = platform_get_transactions(platform=platform, league_id=league_id, week=w, season=season)
        return w, tx if isinstance(tx, list) else []

    with ThreadPoolExecutor(max_workers=min(len(season_weeks), 8)) as pool:
        futures = {pool.submit(_fetch, w): w for w in season_weeks}
        for fut in as_completed(futures):
            w = futures[fut]
            try:
                week, tx = fut.result()
                results[week] = tx
            except Exception as e:
                print(f"[transactions] Week {w} failed → {e}")
                results[w] = []

    return results


def build_week_activity(
        league_id: str,
        platform,
        season,
        players_map: Optional[Dict[str, Dict[str, str]]] = None,
) -> pd.DataFrame:
    """
    Builds a season-long activity table with:
        kind: 'trade' | 'waiver'
        week: int
        ts: datetime (UTC)
        data: structured payload for HTML
    Optimized to minimize repeated lookups and work.
    """

    # You can still change this to a dynamic list if needed
    season_weeks = list(range(1, 19))

    roster_name, roster_avatar = build_roster_display_maps(league_id, platform, season)
    tx_by_week = get_transactions_by_week(league_id, season_weeks, platform=platform, season=int(season)) or {}
    rows: list[dict] = []

    # Fast path: no transactions at all
    if not tx_by_week:
        return pd.DataFrame(columns=["kind", "week", "ts", "data"])

    players_map = players_map or {}
    pmap_get = players_map.get
    rows_append = rows.append

    # Memoize pinfo per player id so we don't rebuild dicts repeatedly
    pinfo_cache: Dict[str, Dict[str, Any]] = {}

    def pinfo(pid: str) -> dict[str, Any]:
        pid_str = str(pid)
        cached = pinfo_cache.get(pid_str)
        if cached is not None:
            return cached

        p = pmap_get(pid_str) or {}
        gp = p.get
        info = {
            "name": gp("name", pid_str),
            "pos": gp("pos", ""),
            "team": gp("team", "FA"),
            "age": gp("age", None),
            "pid": pid_str,
        }
        pinfo_cache[pid_str] = info
        return info

    # Iterate only over weeks that actually have transactions
    for week, txs in tx_by_week.items():
        if not txs:
            continue

        for t in txs:
            ttype = t.get("type")

            # Compute timestamp once, cheaply
            ts_raw = t.get("status_updated") or t.get("created")
            if ts_raw:
                ts = datetime.fromtimestamp(ts_raw / 1000.0, tz=timezone.utc)
            else:
                ts = None

            # ---------- WAIVERS ----------
            if ttype in ("waiver", "waiver_add", "free_agent"):
                # Handle Sleeper's free_agent format
                if ttype == "free_agent":
                    adds = t.get("adds")
                    # Only process actual waiver adds, not drops
                    if not isinstance(adds, dict) or not adds:
                        continue

                    by_rid: dict[str, list[dict]] = defaultdict(list)
                    for pid, rid in adds.items():
                        by_rid[str(rid)].append(pinfo(pid))
                else:
                    # Original format for other platforms
                    adds = t.get("adds")
                    if not isinstance(adds, dict) or not adds:
                        continue

                    by_rid: dict[str, list[dict]] = defaultdict(list)
                    for pid, rid in adds.items():
                        by_rid[str(rid)].append(pinfo(pid))

                for rid, players in by_rid.items():
                    rows_append(
                        {
                            "kind": "waiver",
                            "week": week,
                            "ts": ts,
                            "data": {
                                "rid": rid,
                                "name": roster_name.get(rid, f"Roster {rid}"),
                                "avatar": roster_avatar.get(rid),
                                "adds": players,
                            },
                        }
                    )
                continue

            # ---------- TRADES ----------
            if ttype == "trade":
                adds = t.get("adds") or {}
                drops = t.get("drops") or {}
                draft_picks = t.get("draft_picks") or []

                # If absolutely nothing happened, skip
                if not adds and not drops and not draft_picks:
                    continue

                # Collect all team IDs involved in this trade
                base_rosters = set(map(str, t.get("roster_ids") or []))
                rec_teams = {str(v) for v in adds.values()} if adds else set()
                send_teams = {str(v) for v in drops.values()} if drops else set()

                team_ids = base_rosters | rec_teams | send_teams
                if not team_ids:
                    continue

                team_objs = []
                team_objs_append = team_objs.append

                for rid in sorted(team_ids):
                    gets = [pinfo(pid) for pid, to_rid in adds.items() if str(to_rid) == rid] if adds else []
                    sends = [pinfo(pid) for pid, from_rid in drops.items() if str(from_rid) == rid] if drops else []

                    try:
                        rid_int = int(rid)
                    except Exception:
                        rid_int = None

                    team_objs_append(
                        {
                            "rid": rid,
                            "roster_id": rid_int,
                            "name": roster_name.get(rid, f"Roster {rid}"),
                            "avatar": roster_avatar.get(rid),
                            "gets": gets,
                            "sends": sends,
                        }
                    )

                rows_append(
                    {
                        "kind": "trade",
                        "week": week,
                        "ts": ts,
                        "data": {
                            "teams": team_objs,
                            "draft_picks": draft_picks,
                        },
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["kind", "week", "ts", "data"])

    df = pd.DataFrame(rows)

    # ts is already datetime, but this makes us robust if any None snuck in
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.sort_values("ts", ascending=False).reset_index(drop=True)
    return df


def compute_week_opponents(matchups_week: Iterable[Dict[str, Any]]) -> List[Tuple[Any, Any]]:
    if isinstance(matchups_week, dict):
        matchups_week = [matchups_week]

    pairs: List[Tuple[Any, Any]] = []
    matchups_list = list(matchups_week)

    new_shape = any(("left" in m and "right" in m) for m in matchups_list)

    if new_shape:
        for m in matchups_list:
            if "left" not in m or "right" not in m:
                continue
            L = m["left"] or {}
            R = m["right"] or {}
            # Prefer owner display name — team_strength / standings are keyed by it.
            a = L.get("name") or L.get("username") or L.get("roster_id")
            b = R.get("name") or R.get("username") or R.get("roster_id")
            if a is not None and b is not None:
                pairs.append((a, b))
        return pairs

    by_id: Dict[Any, List[Any]] = {}
    for m in matchups_list:
        mid = m.get("matchup_id")
        rid = m.get("roster_id")
        if mid is None or rid is None:
            continue
        by_id.setdefault(mid, []).append(rid)

    for rids in by_id.values():
        if len(rids) == 2:
            pairs.append((rids[0], rids[1]))

    return pairs


def owner_pairs_from_weekly(df: pd.DataFrame) -> List[Tuple[int, str, str]]:
    """(week, owner_a, owner_b) for each head-to-head in a weekly scores frame.

    Uses the same owner keys as team_stats, so past SOS does not depend on
    matchup preview identity fields matching.
    """
    if df is None or getattr(df, "empty", True):
        return []
    if "owner" not in df.columns or "week" not in df.columns:
        return []

    pairs: List[Tuple[int, str, str]] = []

    def _unique_owners(values) -> List[str]:
        seen: List[str] = []
        for raw in values:
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                continue
            owner = str(raw)
            if owner and owner not in seen:
                seen.append(owner)
        return seen

    if "matchup_id" in df.columns:
        for (week, _mid), grp in df.groupby(["week", "matchup_id"], sort=False):
            owners = _unique_owners(grp["owner"].tolist())
            if len(owners) != 2:
                continue
            try:
                w = int(week)
            except (TypeError, ValueError):
                continue
            pairs.append((w, owners[0], owners[1]))
        return pairs

    if "opponent" in df.columns:
        seen_keys: set[tuple] = set()
        for _, row in df.iterrows():
            owners = _unique_owners([row.get("owner"), row.get("opponent")])
            if len(owners) != 2:
                continue
            try:
                w = int(row["week"])
            except (TypeError, ValueError):
                continue
            key = (w, tuple(sorted(owners)))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            pairs.append((w, owners[0], owners[1]))
    return pairs


def _norm_series(series) -> pd.Series:
    s = pd.Series(series, dtype="float64")
    min_v, max_v = float(s.min()), float(s.max())
    if not np.isfinite(min_v) or not np.isfinite(max_v) or max_v == min_v:
        return pd.Series(0.5, index=s.index)
    return (s - min_v) / (max_v - min_v)


def build_team_strength(team_stats: pd.DataFrame) -> dict[str, float]:
    """0–1 opponent weights for SOS.

    Uses season scoring (AVG) blended with win rate — more stable than
    PowerScore, which already mixes Last3 recency and would make schedule
    difficulty chase hot/cold streaks.
    """
    if team_stats is None or getattr(team_stats, "empty", True):
        return {}

    parts: List[pd.Series] = []
    weights: List[float] = []
    if "AVG" in team_stats.columns:
        parts.append(_norm_series(team_stats["AVG"].fillna(0.0)))
        weights.append(0.65)
    if "Win%" in team_stats.columns:
        parts.append(_norm_series(team_stats["Win%"].fillna(0.0)))
        weights.append(0.35)
    if not parts and "PowerScore" in team_stats.columns:
        parts.append(_norm_series(team_stats["PowerScore"].astype(float)))
        weights.append(1.0)
    if not parts:
        parts.append(pd.Series(0.5, index=team_stats.index))
        weights.append(1.0)

    wsum = sum(weights) or 1.0
    blended = sum(w * p for w, p in zip(weights, parts)) / wsum

    strength_by_owner: dict[str, float] = {}
    if "owner" not in team_stats.columns:
        return strength_by_owner
    for idx in team_stats.index:
        owner = team_stats.at[idx, "owner"]
        if owner is None or (isinstance(owner, float) and np.isnan(owner)):
            continue
        strength_by_owner[str(owner)] = float(blended.loc[idx])
    return strength_by_owner


def _sos_alias_map(
        team_strength: Dict[str, float],
        users: Any,
        all_matchups: Optional[Dict[int, List[dict]]] = None,
) -> Dict[str, str]:
    """Map roster_id / username / display_name / team_name → owner key."""
    aliases: Dict[str, str] = {}
    owner_set = set(team_strength)
    owner_lower = {str(o).lower(): o for o in owner_set}

    def _bind(alias: Any, owner: str) -> None:
        if alias is None or owner not in owner_set:
            return
        key = str(alias).strip()
        if not key:
            return
        aliases[key] = owner
        aliases[key.lower()] = owner

    for owner in owner_set:
        _bind(owner, owner)

    for week_ms in (all_matchups or {}).values():
        if isinstance(week_ms, dict):
            week_ms = [week_ms]
        for m in week_ms or []:
            if not isinstance(m, dict):
                continue
            for side in ("left", "right"):
                team = m.get(side) or {}
                name = team.get("name")
                owner = name if name in owner_set else owner_lower.get(str(name or "").lower())
                if not owner:
                    continue
                _bind(team.get("roster_id"), owner)
                _bind(team.get("username"), owner)
                _bind(name, owner)

    if isinstance(users, dict):
        user_iter = users.values()
    else:
        user_iter = users or []
    for u in user_iter:
        if not isinstance(u, dict):
            continue
        team_name = (u.get("metadata") or {}).get("team_name") or u.get("display_name")
        owner = None
        for cand in (team_name, u.get("display_name"), u.get("username")):
            if cand in owner_set:
                owner = cand
                break
            if cand is not None:
                owner = owner_lower.get(str(cand).lower())
                if owner:
                    break
        if not owner:
            continue
        _bind(u.get("display_name"), owner)
        _bind(u.get("username"), owner)
        _bind(team_name, owner)

    return aliases


def _sos_accumulate(
        out: dict,
        team_strength: Dict[str, float],
        a: Optional[str],
        b: Optional[str],
        bucket: str,
) -> None:
    if not a or not b or a == b or a not in out or b not in out:
        return
    out[a][f"{bucket}_sos"] += team_strength[b]
    out[a][f"{bucket}_cnt"] += 1
    out[b][f"{bucket}_sos"] += team_strength[a]
    out[b][f"{bucket}_cnt"] += 1


def _sos_indexify(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mu = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / len(values)
    return mu, var ** 0.5


def compute_sos_by_team(
        all_matchups: Dict[int, List[dict]],
        team_strength: Dict[str, float],
        weeks_past: int,
        users: Any = None,
        regular_season_weeks: int = 14,
        past_pairs: Optional[Iterable[tuple]] = None,
) -> Dict[str, dict]:
    """Past / rest-of-season SOS indexed to 100 = league average, +10 per σ.

    Higher = tougher opponents. Regular-season only: playoff weeks in
    ``all_matchups`` are ignored. ``past_pairs`` (from weekly scores) is the
    preferred source for games already played because those rows share owner
    keys with ``team_strength``.
    """
    out: dict[str, dict[str, Any]] = {
        str(owner): {"past_sos": 0.0, "past_cnt": 0, "ros_sos": 0.0, "ros_cnt": 0}
        for owner in team_strength
    }
    strength = {str(k): float(v) for k, v in team_strength.items()}
    aliases = _sos_alias_map(strength, users, all_matchups)

    def _resolve(token: Any) -> Optional[str]:
        if token is None:
            return None
        key = str(token)
        if key in aliases:
            return aliases[key]
        return aliases.get(key.lower())

    try:
        past_end = min(max(0, int(weeks_past)), int(regular_season_weeks))
        ros_end = int(regular_season_weeks)
    except (TypeError, ValueError):
        past_end, ros_end = 0, 14

    used_past_pairs = False
    if past_pairs is not None:
        for item in past_pairs:
            if item is None:
                continue
            week = None
            if len(item) >= 3:
                week, a_raw, b_raw = item[0], item[1], item[2]
            elif len(item) == 2:
                a_raw, b_raw = item
            else:
                continue
            if week is not None:
                try:
                    w = int(week)
                except (TypeError, ValueError):
                    continue
                if w < 1 or w > past_end:
                    continue
            a, b = _resolve(a_raw), _resolve(b_raw)
            _sos_accumulate(out, strength, a, b, "past")
            used_past_pairs = True

    if not used_past_pairs:
        for w in range(1, past_end + 1):
            for a_raw, b_raw in compute_week_opponents((all_matchups or {}).get(w, [])):
                _sos_accumulate(out, strength, _resolve(a_raw), _resolve(b_raw), "past")

    matchups = all_matchups or {}
    for w in range(past_end + 1, ros_end + 1):
        for a_raw, b_raw in compute_week_opponents(matchups.get(w, [])):
            _sos_accumulate(out, strength, _resolve(a_raw), _resolve(b_raw), "ros")

    past_vals: list[float] = []
    ros_vals: list[float] = []
    for v in out.values():
        if v["past_cnt"]:
            v["past_sos"] = v["past_sos"] / v["past_cnt"]
            past_vals.append(v["past_sos"])
        else:
            v["past_sos"] = 0.0

        if v["ros_cnt"]:
            v["ros_sos"] = v["ros_sos"] / v["ros_cnt"]
            ros_vals.append(v["ros_sos"])
        else:
            v["ros_sos"] = 0.0

    mu_p, sigma_p = _sos_indexify(past_vals)
    mu_r, sigma_r = _sos_indexify(ros_vals)

    for v in out.values():
        if v["past_cnt"] and sigma_p > 0:
            v["past_sos"] = 100.0 + 10.0 * (v["past_sos"] - mu_p) / sigma_p
        elif v["past_cnt"]:
            v["past_sos"] = 100.0
        else:
            v["past_sos"] = 0.0

        if v["ros_cnt"] and sigma_r > 0:
            v["ros_sos"] = 100.0 + 10.0 * (v["ros_sos"] - mu_r) / sigma_r
        elif v["ros_cnt"]:
            v["ros_sos"] = 100.0
        else:
            v["ros_sos"] = 0.0

    return out


from collections import defaultdict


def playoff_bracket(
        winners_bracket,
        roster_name_map,
        roster_avatar_map,
        match_scores=None,
        seed_map=None,
):
    if not winners_bracket:
        return "<div class='po-empty'>No playoff bracket available.</div>"

    match_scores = match_scores or {}
    seed_map = seed_map or {}

    def _k(x):
        return str(x) if x is not None else None

    def seed_key(rid):
        if rid is None:
            return 9999
        s = seed_map.get(str(rid))
        if s is None and isinstance(rid, (int, str)) and str(rid).isdigit():
            s = seed_map.get(int(rid), None)
        try:
            rid_int = int(rid)
        except Exception:
            rid_int = 9999
        return (s if s is not None else 9999, rid_int)

    roster_name = {_k(k): v for k, v in (roster_name_map or {}).items()}

    # --- figure out who is in the playoffs + byes ---------------------------
    all_playoff_rids = set()
    round1_rids = set()
    for m in winners_bracket:
        r = m.get("r")
        for key in ("t1", "t2"):
            rid = m.get(key)
            if rid is not None:
                all_playoff_rids.add(rid)
                if r == 1:
                    round1_rids.add(rid)

    # use this BEFORE we inject bye matches so we know "real" field size
    playoff_field_size = len(all_playoff_rids)

    bye_rids = all_playoff_rids - round1_rids
    bye_rids_sorted = sorted(bye_rids, key=seed_key)

    extended_bracket = list(winners_bracket)
    round1_override = None

    if bye_rids_sorted:
        existing_ids = [m.get("m") for m in winners_bracket if isinstance(m.get("m"), int)]
        next_m = max(existing_ids) + 1 if existing_ids else 1

        bye_matches = []
        for rid in bye_rids_sorted:
            bye_matches.append(
                {
                    "m": next_m,
                    "r": 1,
                    "w": None,
                    "l": None,
                    "t1": rid,
                    "t2": None,
                    "t1_from": None,
                    "t2_from": None,
                    "is_bye": True,
                }
            )
            next_m += 1

        r1_existing = [m for m in extended_bracket if m.get("r") == 1 and not m.get("is_bye")]
        non_r1 = [m for m in extended_bracket if m.get("r") != 1]

        if len(bye_matches) == 1:
            new_r1 = bye_matches[:1] + r1_existing
        elif len(bye_matches) >= 2:
            middle_byes = bye_matches[1:-1]
            new_r1 = [bye_matches[0]] + r1_existing + middle_byes + [bye_matches[-1]]
        else:
            new_r1 = r1_existing

        extended_bracket = non_r1 + new_r1
        round1_override = new_r1

    winners_bracket = extended_bracket
    match_by_id = {m["m"]: m for m in winners_bracket if "m" in m}

    rounds: dict[int, list] = defaultdict(list)
    for m in winners_bracket:
        r = m.get("r")
        if r is None:
            continue
        rounds[r].append(m)

    if not rounds:
        return "<div class='po-empty'>No playoff bracket available.</div>"

    if round1_override:
        rounds[1] = round1_override

    round_nums = sorted(rounds.keys())
    for r in round_nums:
        if r == 1 and round1_override:
            continue
        rounds[r].sort(key=lambda x: x.get("m", 0))

    # --------- dynamic round labels based on size / number of rounds --------
    total_rounds = max(round_nums)

    def get_round_label(r: int) -> str:
        """
        Make the labels adapt to:
        - 4-team playoffs (2 rounds): Semifinals -> Finals
        - 6-team playoffs (3 rounds): Round 1 -> Semifinals -> Finals
        - 8-team playoffs (3 rounds): Quarterfinals -> Semifinals -> Finals
        - bigger fields: generic + last two rounds as Semis/Finals.
        """
        if total_rounds == 1:
            # just a championship game
            return "Finals"

        # final is always "Finals"
        if r == total_rounds:
            return "Finals"

        if total_rounds == 2:
            # 4-team bracket: R1 = Semis, R2 = Finals
            return "Semifinals"

        if total_rounds == 3:
            # 3-round bracket: handle 6-team vs 8-team naming
            if r == total_rounds - 1:
                return "Semifinals"
            # earliest round:
            if playoff_field_size >= 8:
                return "Quarterfinals"
            else:
                return "Round 1"

        # 4+ rounds: last two named, earlier ones generic / quarters
        if r == total_rounds - 1:
            return "Semifinals"
        if r == total_rounds - 2:
            return "Quarterfinals"

        return f"Round {r}"

    # ---------------------- slot + HTML rendering ---------------------------
    def resolve_slot(match, side_key):
        rid = match.get(side_key)
        from_spec = match.get(f"{side_key}_from")

        if rid is not None:
            key = _k(rid)
            return {
                "label": roster_name.get(key, f"Roster {key}"),
                "avatar": roster_avatar_map.get(roster_name.get(key, "")),
                "kind": "team",
                "roster_id": str(rid),
            }

        if isinstance(from_spec, dict) and from_spec:
            src_type, src_mid = next(iter(from_spec.items()))
            if src_type == "l":
                return None
            if src_type == "w":
                src = match_by_id.get(src_mid, {})
                t1_rid = src.get("t1")
                t2_rid = src.get("t2")
                team1 = roster_name.get(_k(t1_rid)) if t1_rid is not None else None
                team2 = roster_name.get(_k(t2_rid)) if t2_rid is not None else None
                if not team1 or not team2:
                    return {"label": "TBD", "avatar": "", "kind": "empty"}
                return {"label": f"{team1}/{team2}", "avatar": "", "kind": "from"}

        other = "t2" if side_key == "t1" else "t1"
        if match.get(other) is not None or match.get(f"{other}_from"):
            return {"label": "BYE", "avatar": "", "kind": "bye"}

        return {"label": "TBD", "avatar": "", "kind": "empty"}

    def render_team_row(slot, score_text, top=False):
        cls = "team-row"
        if slot["kind"] == "bye":
            cls += " bye"
        if top:
            cls += " top"

        if slot.get("avatar"):
            img = (
                f"<div class='team-avatar'><img src='{slot['avatar']}' "
                "onerror=\"this.style.display='none'\"></div>"
            )
        else:
            img = "<div class='team-avatar'></div>"

        rid = slot.get("roster_id")
        clickable_attrs = (
            f" class='team-name team-clickable' data-roster-id='{rid}' data-team-name='{slot['label']}'"
            if rid else " class='team-name'"
        )

        return (
            f"<div class='{cls}'>"
            f"  <div class='team-main'>"
            f"    {img}"
            f"    <div class='team-text'><div{clickable_attrs}>{slot['label']}</div></div>"
            f"  </div>"
            f"  <div class='team-score'>{score_text}</div>"
            f"</div>"
        )

    html_rounds = []
    for r in round_nums:
        round_label = get_round_label(r)
        matches = rounds[r]
        match_html = []
        for m in matches:
            mid = m.get("m")
            scores = match_scores.get(mid, {}) if mid is not None else {}

            slot1 = resolve_slot(m, "t1")
            slot2 = resolve_slot(m, "t2")

            if slot1 is None or slot2 is None:
                continue

            s1 = scores.get("t1_score")
            s2 = scores.get("t2_score")
            s1_txt = f"{s1:.2f}" if isinstance(s1, (int, float)) else "–"
            s2_txt = f"{s2:.2f}" if isinstance(s2, (int, float)) else "–"

            if slot1["kind"] == "bye":
                s1_txt = "-"
            if slot2["kind"] == "bye":
                s2_txt = "-"

            match_html.append(
                "<div class='bracket-match'>"
                f"  {render_team_row(slot1, s1_txt, top=True)}"
                f"  {render_team_row(slot2, s2_txt, top=False)}"
                "</div>"
            )

        if match_html:
            html_rounds.append(
                f"<div class='bracket-round round-{r}'>"
                f"  <div class='round-title'>{round_label}</div>"
                f"  <div class='round-body'>{''.join(match_html)}</div>"
                f"</div>"
            )

    if not html_rounds:
        return "<div class='po-empty'>No playoff bracket available.</div>"

    # data-br-moment="bracket": the rounds build in left-to-right when the
    # bracket first scrolls into view (see the big-moment CSS in dashboard.css).
    return "<div class='bracket' data-br-moment='bracket'>" + "".join(html_rounds) + "</div>"


def seed_top_n_from_team_stats(team_stats, roster_map, playoff_size: int = 6):
    """
    Build a seed_map {roster_id: seed} for the top N teams (N = playoff_size)
    based on Wins, PF, PA, owner name tiebreak.

    This works for:
      - 4-team playoffs (playoff_size=4)
      - 6-team (playoff_size=6)
      - 8-team (playoff_size=8)
      - etc.
    """
    if playoff_size <= 0:
        raise ValueError("playoff_size must be positive")

    required_cols = {"owner", "Wins", "PF", "PA"}
    missing = required_cols - set(team_stats.columns)
    if missing:
        raise ValueError(f"team_stats missing required columns: {missing}")
    if not isinstance(roster_map, dict) or not roster_map:
        raise ValueError("roster_map must be a non-empty dict of {roster_id: team_name}")

    # Map normalized owner name -> roster_id
    name_to_rid: dict[str, str] = {}
    for rid, name in roster_map.items():
        if isinstance(name, str):
            key = name.strip()
            if key:
                name_to_rid[key] = str(rid).strip()

    df = team_stats.copy()
    df["owner_norm"] = df["owner"].astype(str).str.strip()

    df_sorted = (
        df.sort_values(
            by=["Wins", "PF", "PA", "owner_norm"],
            ascending=[False, False, True, True],
        )
        .reset_index(drop=True)
    )

    seed_map: dict[str, int] = {}
    seed = 1
    for _, row in df_sorted.iterrows():
        owner_name = row["owner_norm"]
        rid = name_to_rid.get(owner_name)
        if not rid:
            continue
        if rid in seed_map:
            continue
        seed_map[rid] = seed
        seed += 1
        if seed > playoff_size:
            break

    return seed_map


# Backwards-compatible wrapper if you still use the old function name anywhere
def seed_top6_from_team_stats(team_stats, roster_map):
    return seed_top_n_from_team_stats(team_stats, roster_map, playoff_size=6)


def render_teams_sidebar(teams: List[dict]) -> str:
    if not teams:
        return ""

    pill_buttons = []
    for idx, t in enumerate(teams):
        active_class = " active" if idx == 0 else ""
        label = html.escape(t.get("name") or t.get("username") or f"Team {t['roster_id']}")
        pill_buttons.append(
            f"<button class='manager-pill{active_class}' "
            f"data-team-id='{t['roster_id']}'>{label}</button>"
        )
    header_html = (
            "<div class='manager-pills-carousel'>"
            "<button class='pill-arrow pill-arrow-left' type='button'>&lsaquo;</button> "
            "<div class='manager-pills-row'>"
            + "".join(pill_buttons)
            + "</div><button class='pill-arrow pill-arrow-right' type='button'>&rsaquo;</button></div>"
    )

    panel_html_parts = []

    for idx, t in enumerate(teams):
        active_class = " active" if idx == 0 else ""

        def render_player_list(title: str, players: List[dict], extra_class: str = "") -> str:
            out: list[str] = []
            out.append("<div class='team-section'>")
            out.append(f"<div class='team-section-title'>{title}</div>")
            out.append("<div class='player-list'>")
            if players:
                for p in players:
                    row_cls = "player-row"
                    if p.get("name") == "0":
                        p['name'] = "Empty"
                        row_cls = "player-row empty"
                    if extra_class:
                        row_cls += f" {extra_class}"
                    pos = p.get("pos")
                    pos_badge = f"<span class='pos-badge {pos}'>{pos}</span>" if pos else ""
                    nfl = p.get("nfl")
                    pos = p.get("pos")
                    nfl_html = f"<span class='meta'>{nfl}</span>" if nfl and nfl != "FA" else ""

                    # Make player name clickable; unknown/DEF players are not clickable
                    pid = p.get("pid", "")
                    player_name = p['name']
                    is_unknown = not player_name or player_name in ("Unknown", "0", "Empty") or str(player_name).isdigit()
                    if pos == "DEF" and nfl:
                        city = _team_city(nfl)
                        player_name = player_name.replace(nfl, city) if nfl in player_name else city
                        clickable_attrs = " class='pname'"
                    elif is_unknown or pid == "0":
                        player_name = f"Unknown {pos}" if pos and str(player_name).isdigit() else (player_name or "Unknown")
                        clickable_attrs = " class='pname' style='color:var(--text-muted);'"
                    else:
                        clickable_attrs = f" class='pname player-clickable' style='cursor:pointer;' data-player-id='{pid}' data-player-name='{player_name}'"

                    out.append(
                        f"<div class='{row_cls}'>"
                        f"{pos_badge}"
                        f"<span{clickable_attrs}>{player_name}</span>"
                        f"{nfl_html}"
                        "</div>"
                    )
            else:
                out.append("<div class='player-row empty'>None</div>")
            out.append("</div></div>")
            return "".join(out)

        sections = []
        if t["starters"]:
            sections.append(render_player_list("Starters", t["starters"]))
        if t["bench"]:
            sections.append(render_player_list("Bench", t["bench"]))
        if t.get("ir"):
            sections.append(render_player_list("IR", t["ir"], extra_class="ir"))
        if t["taxi"]:
            sections.append(render_player_list("Taxi", t["taxi"], extra_class="taxi"))

        picks = t.get("picks") or []
        picks_out: list[str] = []
        if picks:
            picks_out.append("<div class='team-section'>")
            picks_out.append("<div class='team-section-title'>Picks</div>")
            picks_out.append("<div class='player-list picks-list'>")
            for pk in picks:
                season = pk.get("season", "")
                rnd = pk.get("round", "")
                via = pk.get("original_owner")
                via_txt = f" (via {via})" if via else ""
                picks_out.append(
                    f"<div class='pick-row'>{season} • Round {rnd}{via_txt}</div>"
                )
            picks_out.append("</div></div>")

        body_html = (
                "<div class='team-body'>"
                + "".join(sections)
                + "".join(picks_out)
                + "</div>"
        )

        panel_html_parts.append(
            f"<div class='team-panel{active_class}' data-team-id='{t['roster_id']}'>"
            f"{body_html}"
            "</div>"
        )

    panels_html = "<div class='team-panels'>" + "".join(panel_html_parts) + "</div>"

    card_html = (
        "<div class='card teams-card' data-section='overview'>"
        f"{header_html}"
        f"{panels_html}"
        "</div>"
    )
    return card_html


def render_predraft_sidebar(
        platform: str,
        season,
        league_id: str,
        preview: Optional[List[dict]] = None,
) -> str:
    """Roster-sidebar stand-in when the league has not drafted yet.

    Empty team shells used to render an empty manager-pill card. Replace that
    with the cheat sheet (preview + link) so the right column is useful.
    """
    plat = html.escape(str(platform or "sleeper"))
    seas = html.escape(str(season or ""))
    lid = html.escape(str(league_id or ""))
    sheet_href = f"/{plat}/{seas}/{lid}/draft/cheat-sheet"
    room_href = f"/{plat}/{seas}/{lid}/draft"

    rows = []
    for i, p in enumerate(preview or [], start=1):
        pos = html.escape(str(p.get("pos") or ""))
        name = html.escape(str(p.get("name") or "Player"))
        pid = html.escape(str(p.get("id") or ""))
        if pid:
            name_html = (
                f"<span class='pname player-clickable' style='cursor:pointer;' "
                f"data-player-id='{pid}' data-player-name='{name}'>{name}</span>"
            )
        else:
            name_html = f"<span class='pname'>{name}</span>"
        pos_badge = f"<span class='pos-badge {pos}'>{pos}</span>" if pos else ""
        rows.append(
            f"<div class='player-row'>"
            f"<span class='os-draft-prep-rk'>{i}</span>"
            f"{pos_badge}{name_html}"
            f"</div>"
        )
    list_html = (
        f"<div class='player-list os-draft-prep-list'>{''.join(rows)}</div>"
        if rows else
        "<p class='os-draft-prep-empty'>Open the cheat sheet to rank this league's board.</p>"
    )
    return (
        "<div class='card teams-card os-draft-prep' data-section='draft-prep'>"
        "<div class='os-section-head'>"
        "<div class='os-section-head-content'>"
        "<h2 class='os-section-title'>Draft Cheat Sheet</h2>"
        "<div class='os-section-subtitle'>Rosters aren't set yet. Rank the board before you draft.</div>"
        "</div></div>"
        f"{list_html}"
        "<div class='os-draft-prep-actions'>"
        f"<a class='os-draft-prep-primary' href='{sheet_href}'>Open cheat sheet</a>"
        f"<a class='os-draft-prep-secondary' href='{room_href}'>Draft Room</a>"
        "</div></div>"
    )


def render_dashboard_teams_sidebar(
        ctx: dict,
        teams: List[dict],
        filled_label: str = "Roster",
):
    """Roster sidebar, or the cheat-sheet stand-in when the league is undrafted.

    Returns ``(html, jump_nav_label)``.
    """
    from utils.league_payload import startup_draft_pending, top_board_preview
    from utils.lineup_slots import canonicalize_slot

    rosters = (ctx or {}).get("rosters") or []
    if startup_draft_pending(
        (ctx or {}).get("league"),
        (ctx or {}).get("latest_draft"),
        rosters,
    ):
        rp = (ctx or {}).get("roster_positions") or []
        is_sf = any(canonicalize_slot(s) == "SUPER_FLEX" for s in rp)
        html_out = render_predraft_sidebar(
            platform=(ctx or {}).get("platform") or "sleeper",
            season=(ctx or {}).get("season") or (ctx or {}).get("current_season"),
            league_id=str((ctx or {}).get("league_id") or ""),
            preview=top_board_preview(
                (ctx or {}).get("model_value_table") or [], is_sf=is_sf,
            ),
        )
        return html_out, "Cheat Sheet"
    return render_teams_sidebar(teams), filled_label


def build_picks_by_roster(
        num_future_seasons: int = 3,
        league: dict = None,
        rosters: List[dict] = None,
        traded: List[dict] = None,
        draft_ended: bool = False,
) -> Dict[str, List[dict]]:
    current_season = int((league or {}).get("season") or 0)
    if not current_season:
        return {}
    num_rounds = int((league or {}).get("settings", {}).get("draft_rounds", 4))
    start_offset = 1 if draft_ended else 0

    all_picks: List[dict] = []
    roster_ids = [int(r["roster_id"]) for r in rosters]

    for offset in range(start_offset, start_offset + num_future_seasons):
        season = current_season + offset
        for rid in roster_ids:
            for rnd in range(1, num_rounds + 1):
                all_picks.append(
                    {
                        "season": season,
                        "round": rnd,
                        "original_roster_id": rid,
                        "owner_roster_id": rid,
                    }
                )

    traded = traded or []
    for tp in traded:
        try:
            season = int(tp["season"])
            rnd = int(tp["round"])
            original = int(tp["roster_id"])
            new_owner = int(tp["owner_id"])
        except (KeyError, ValueError, TypeError):
            continue

        for p in all_picks:
            if (
                    p["season"] == season
                    and p["round"] == rnd
                    and p["original_roster_id"] == original
            ):
                p["owner_roster_id"] = new_owner

    picks_by_roster: Dict[str, List[dict]] = {}
    for p in all_picks:
        owner_key = str(p["owner_roster_id"])
        picks_by_roster.setdefault(owner_key, []).append(
            {
                "season": p["season"],
                "round": p["round"],
                "original_owner": str(p["original_roster_id"]),
            }
        )

    for rid in picks_by_roster:
        picks_by_roster[rid].sort(key=lambda x: (x["season"], x["round"]))

    return picks_by_roster


def age_from_bday(bday: Optional[str]) -> Optional[float]:
    if not bday:
        return None
    try:
        parts = bday.split("T")[0].split("/")
        month, day, year = map(int, parts[:3])
        dob = date(year, month, day)
        as_of = date.today()
        days = (as_of - dob).days
        age = days / 365.25
        # Floor-truncate to 1 decimal so a player 4 days from their 30th birthday
        # shows 29.9, not 30.0 (round() would push 29.98 → 30.0).
        import math
        return math.floor(age * 10) / 10
    except Exception:
        return None


def pill(s):
    return f"<span class='badge'>{s}</span>"


def build_standings_map(team_stats, roster_map) -> dict[int, int]:
    ordered = (
        team_stats.sort_values(["Wins", "PF"], ascending=[False, False]).reset_index(drop=True)
    )
    owner_to_rid = {owner: rid for rid, owner in roster_map.items()}

    standings: dict[int, int] = {}
    for idx, row in ordered.iterrows():
        owner = row["owner"]
        rid = owner_to_rid.get(owner)
        seed = idx + 1
        standings[rid] = seed
    return standings
