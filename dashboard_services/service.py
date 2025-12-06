from __future__ import annotations

import json
import re
import numpy as np
import pandas as pd
import requests
import time
from bs4 import BeautifulSoup
from collections import defaultdict
from datetime import datetime, timezone, date
from pathlib import Path
from plotly.offline import plot as plotly_plot
from typing import Dict, Any, Iterable, Tuple, Optional, List, Union, Callable
from zoneinfo import ZoneInfo

from dashboard_services.api import (
    get_matchups,
    _avatar_url,
    get_nfl_state,
    avatar_from_users,
    get_transactions,
    fetch_team_game_logs_html,
)
from dashboard_services.data_building.value_model_training import normalize_name
from dashboard_services.matchups import build_matchup_preview
from dashboard_services.players import build_roster_display_maps
from dashboard_services.styles import recap_css, tickerCss
from dashboard_services.utils import safe_owner_name, path_week_stats, write_json


def render_weekly_highlight_ticker(high: dict, week: int) -> str:
    if not high:
        return ""

    def item(label, value):
        return (
            "<div class='tick-item'>"
            f"  <span class='tick-label'>{label}</span>"
            f"  <span class='tick-val'>{value}</span>"
            "</div>"
        )

    items = []
    tt = high.get("top_team")
    if tt:
        items.append(item("Highest Scoring Team", f"{tt[0]} — {tt[1]}"))

    lt = high.get("low_team")
    if lt:
        items.append(item("Lowest Scoring Team", f"{lt[0]} — {lt[1]}"))

    cl = high.get("closest")
    if cl:
        a, b, diff, pa, pb = cl
        items.append(item("Closest Matchup", f"{a} {pa} – {pb} {b} (Δ{diff:.2f})"))

    bl = high.get("blowout")
    if bl:
        a, b, diff, pa, pb = bl
        items.append(item("Biggest Blowout", f"{a} {pa} – {pb} {b} (Δ{diff:.2f})"))

    loop = "".join(items)
    track_html = loop + loop if items else "<div class='tick-item'>No highlights</div>"

    return f"""
    <div class="grid tick-wrap ticker" aria-label="Week {week} Highlights" data-section="recap">
      <div class="tick-head">Week {week} Highlights</div>
      <div class="tick-viewport">
        <div class="tick-track">{track_html}</div>
      </div>
      {tickerCss}
    </div>
    """


def matchup_cards_last_week(
    league_id: str,
    df_weekly: pd.DataFrame,
    roster_map: dict,
    players_map: dict,
    rosters: list,
    users: list,
) -> tuple[int, str, dict]:
    """
    Returns: (week_number, html_for_matchup_cards, top_by_pos_dict)
      top_by_pos_dict: {'QB': [ {name, pts, nfl, team, owner}, ... up to 3 ], ...}
    """
    last_week = int(df_weekly["week"].max())
    raw = get_matchups(league_id, last_week) or []

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
            settings.get("wins", 0),
            settings.get("losses", 0),
        )
        owner_id = r.get("owner_id")
        avatar_by_rid[rid] = avatar_from_users(users, owner_id)

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
        if not rows:
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
                <img class="avatar" src="{avatar}" onerror="this.style.display='none'">
                <div class="mu-name left"><div style="display: flex; justify-content: flex-start;">{ln}</div><div style="font-weight: 400; font-size: small;">{winsL}-{lossesL} • @{username}</div></div>
              </div>
              <div class="mu-score">{lp:.2f}</div>
            </div>
            <div class="mu-vs">vs</div>
            <div class="mu-team {r_cls}">
              <div class="mu-score">{rp:.2f}</div>
              <div style="display: flex; align-items: center; justify-content: flex-end; gap: 5px;">
                <div class="mu-name right"><div style="display: flex; justify-content: flex-end;">{rn}</div><div style="font-weight: 400; font-size: small">@{username2} • {winsR}-{lossesR}</div></div>
                <img class="avatar" src="{avatar2}" onerror="this.style.display='none'">
              </div>
            </div>
          </div>
        </div>
        """
        )

    want_positions = ["QB", "RB", "WR", "TE", "K", "DEF"]
    top_by_pos = {}
    for pos in want_positions:
        pool = sorted(buckets.get(pos, []), key=lambda x: x["pts"], reverse=True)[:3]
        top_by_pos[pos] = pool

    return last_week, "".join(cards), top_by_pos


def fantasy_team_for_player(pid: str, rosters: list, roster_map: dict) -> str:
    """
    pid: Sleeper player ID as string
    rosters: list returned by get_rosters()
    roster_map: maps roster_id -> fantasy team name
    """
    for r in rosters:
        if pid in (r.get("players") or []):
            rid = str(r["roster_id"])
            return roster_map.get(rid, f"Roster {rid}")
    return "Free Agent"


def render_top_three(top_by_pos: dict, rosters, roster_map) -> str:
    def card(pos, rows):
        if not rows:
            return f"<div class='side-card'><h2>{pos}</h2><div class='muted'>No data</div></div>"
        lis = []
        for i, r in enumerate(rows, start=1):
            team = r.get("nfl") or r.get("team", "")
            pts = r.get("pts") or r.get("points", 0.0)
            if not r.get("owner_id"):
                owner = fantasy_team_for_player(r["pid"], rosters, roster_map)
            place = "first" if i == 1 else "second" if i == 2 else "third"
            lis.append(
                f"<div class='side-row'>"
                f"  <span class='rank rank-{place}'>{i}</span>"
                f"  <div class='who'>"
                f"    <div class='name'>{r['name']}</div>"
                f"    <div class='sub'>{team} • {owner}</div>"
                f"  </div>"
                f"  <div class='pts'>{pts:.1f}</div>"
                f"</div>"
            )
        return f"<div class='side-card'><h3>{pos}</h3>{''.join(lis)}</div>"

    blocks = [card(pos, top_by_pos.get(pos, [])) for pos in ["QB", "RB", "WR", "TE", "K", "DEF"]]
    return "<div class='sidebar-grid'>" + "".join(blocks) + "</div>"


def render_week_recap_tab(
    league_id: str,
    df_weekly: pd.DataFrame,
    roster_map: dict,
    players_map: dict,
    rosters: list,
    users: list,
) -> str:
    """
    Returns a single <div class='card' data-section='recap'> ... </div> block
    that you can insert into your main page (no new page).
    """
    week, matchup_html, top_by_pos = matchup_cards_last_week(
        league_id, df_weekly, roster_map, players_map, rosters, users
    )

    sidebar_html = render_top_three(top_by_pos)

    return f"""
    <div class="card recap" data-section="recap">
      <div class="recap-grid">
        <h2>Week {week} Recap</h2>
        {matchup_html}
      </div>
    </div>
    <div class="card potw" data-section="recap">
          {sidebar_html}
    </div>
    {recap_css}
    """


def build_tables(
    league_id: str,
    max_week: int,
    players: dict,
    users: list[dict],
    rosters: list[dict],
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

    matchups_by_week = build_matchups_by_week(league_id, range(1, 18), roster_map, players)

    # precompute owner_avatar using user_by_id only (no extra scan over users)
    owner_avatar: dict[str, Union[str, None]] = {}
    for r in rosters:
        rid = str(r["roster_id"])
        owner_id = r.get("owner_id")
        display = roster_map.get(rid, f"Roster {rid}")

        avatar_id = None
        user_data = user_by_id.get(owner_id)
        if user_data:
            user_meta = user_data.get("metadata") or {}
            u_id = user_data.get("avatar")
            avatar_id = user_meta.get("avatar") or (f"https://sleepercdn.com/avatars/{u_id}" if u_id else None)

        owner_avatar[display] = _avatar_url(avatar_id)

    weekly_rows: list[dict] = []
    for week in range(1, max_week + 1):
        try:
            week_data = get_matchups(league_id, week)
        except requests.HTTPError:
            break
        if not week_data:
            continue

        for m in week_data:
            rid = str(m.get("roster_id"))
            weekly_rows.append(
                {
                    "week": week,
                    "matchup_id": m.get("matchup_id"),
                    "roster_id": rid,
                    "owner": roster_map.get(rid, f"Roster {rid}"),
                    "points": float(m.get("points", 0.0)),
                }
            )
        time.sleep(0.12)

    df_weekly = pd.DataFrame(weekly_rows)
    if df_weekly.empty:
        raise SystemExit("No matchup data found. Check league ID and weeks.")

    df_weekly["points_against"] = np.nan
    for (_, _mid), grp in df_weekly.groupby(["week", "matchup_id"]):
        if len(grp) == 2:
            i1, i2 = grp.index.tolist()
            p1, p2 = df_weekly.loc[i1, "points"], df_weekly.loc[i2, "points"]
            df_weekly.loc[i1, "points_against"] = p2
            df_weekly.loc[i2, "points_against"] = p1

    df_weekly["avatar"] = df_weekly["owner"].map(owner_avatar)

    _state = get_nfl_state()
    current_leg = int(_state.get("leg") or _state.get("week") or 0)
    df_weekly["finalized"] = df_weekly["week"] < current_leg

    finalized_mask = df_weekly["finalized"] == True
    df_finalized = df_weekly[finalized_mask].copy()

    records = _compute_team_records(df_finalized.copy())
    team_stats = _aggregate_team_stats(df_finalized.copy(), records)

    team_stats = team_stats.merge(
        pd.Series(owner_avatar, name="avatar"),
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
    cons_inv = -team_stats.get(
        "STD",
        pd.Series(team_stats["STD"].mean(), index=team_stats.index),
    ).fillna(team_stats["STD"].mean())
    ceiling = team_stats.get("MAX", pd.Series(0.0, index=team_stats.index)).fillna(0.0)
    last3_series = team_stats["Last3"].fillna(0.0)

    team_stats["Z_WinPercentage"] = _z(win_pct)
    team_stats["Z_Avg"] = _z(avg_pts)
    team_stats["Z_Last3"] = _z(last3_series)
    team_stats["Z_Consistency"] = _z(cons_inv)
    team_stats["Z_Ceiling"] = _z(ceiling)

    W_WIN, W_AVG, W_LAST3, W_CONS, W_CEIL = 0.2, 0.3, 0.15, 0.20, 0.15
    team_stats["Win%"] = win_pct
    team_stats["PowerScore"] = (
        W_WIN * team_stats["Z_WinPercentage"]
        + W_AVG * team_stats["Z_Avg"]
        + W_LAST3 * team_stats["Z_Last3"]
        + W_CONS * team_stats["Z_Consistency"]
        + W_CEIL * team_stats["Z_Ceiling"]
    )

    sos = build_team_strength(team_stats)
    last_week = int(df_weekly["week"].max())
    sos_dict = compute_sos_by_team(matchups_by_week, sos, last_week, users)
    sos_df = (
        pd.DataFrame.from_dict(sos_dict, orient="index")
        .reset_index()
        .rename(columns={"index": "owner"})
    )
    team_stats = team_stats.merge(sos_df, on="owner", how="left")

    streaks_df = compute_streaks(df_finalized.copy())
    team_stats = team_stats.merge(streaks_df, on="owner", how="left")

    team_stats["StreakType"] = team_stats["StreakType"].fillna("")
    team_stats["StreakLen"] = team_stats["StreakLen"].fillna(0).astype(int)
    team_stats["Streak"] = team_stats["Streak"].fillna("")

    return df_weekly, team_stats, roster_map


def build_league_summary(team_stats, df_weekly) -> dict:
    summary: dict[str, Any] = {}

    if not team_stats.empty:
        best_pf_row = team_stats.loc[team_stats["PF"].idxmax()]
        worst_pf_row = team_stats.loc[team_stats["PF"].idxmin()]
        best_avg_row = team_stats.loc[team_stats["AVG"].idxmax()]
        most_vol_row = team_stats.loc[team_stats["STD"].idxmax()]

        ts = team_stats.copy()
        ts["pf_minus_pa"] = ts["PF"] - ts["PA"]
        luckiest_row = ts.loc[ts["pf_minus_pa"].idxmax()]
        unluckiest_row = ts.loc[ts["pf_minus_pa"].idxmin()]

        summary["best_pf"] = {"owner": best_pf_row["owner"], "pf": float(best_pf_row["PF"])}
        summary["worst_pf"] = {"owner": worst_pf_row["owner"], "pf": float(worst_pf_row["PF"])}
        summary["best_avg"] = {"owner": best_avg_row["owner"], "avg": float(best_avg_row["AVG"])}
        summary["most_vol"] = {"owner": most_vol_row["owner"], "std": float(most_vol_row["STD"])}
        summary["luckiest"] = {
            "owner": luckiest_row["owner"],
            "delta": float(luckiest_row["pf_minus_pa"]),
        }
        summary["unluckiest"] = {
            "owner": unluckiest_row["owner"],
            "delta": float(unluckiest_row["pf_minus_pa"]),
        }

    if not df_weekly.empty:
        best_week_row = df_weekly.loc[df_weekly["points"].idxmax()]
        worst_week_row = df_weekly.loc[df_weekly["points"].idxmin()]

        summary["best_week"] = {
            "owner": best_week_row["owner"],
            "week": int(best_week_row["week"]),
            "points": float(best_week_row["points"]),
        }
        summary["worst_week"] = {
            "owner": worst_week_row["owner"],
            "week": int(worst_week_row["week"]),
            "points": float(worst_week_row["points"]),
        }

    return summary


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
    owners = sorted(set(df["owner"]))
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
    return pd.DataFrame(results)


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

    team_stats["Record"] = team_stats[["Wins", "Losses", "Ties"]].apply(
        lambda r: f"{int(r.Wins)}-{int(r.Losses)}"
        + (f"-{int(r.Ties)}" if r.Ties else ""),
        axis=1,
    )

    return team_stats


def get_owner_id(
    rosters: Optional[list[dict]] = None,
    roster_id: Optional[str] = None,
) -> Optional[str]:
    return next((r["owner_id"] for r in rosters if str(r.get("roster_id")) == str(roster_id)), None)


def build_matchups_by_week(league_id, weeks, roster_map, players_map):
    by_week: dict[int, list] = {}
    for w in weeks:
        matchups = build_matchup_preview(
            league_id=league_id,
            week=w,
            roster_map=roster_map,
            players_map=players_map,
        )
        by_week[w] = matchups or []
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


def get_transactions_by_week(league_id: str, season_weeks: list[int]) -> dict[int, list[dict]]:
    results: dict[int, list[dict]] = {}
    for w in season_weeks:
        try:
            tx = get_transactions(league_id=league_id, week=w)
            results[w] = tx if isinstance(tx, list) else []
        except Exception as e:
            print(f"[transactions] Week {w} failed → {e}")
            results[w] = []
    return results


def build_week_activity(
    league_id: str,
    players_map: Optional[Dict[str, Dict[str, str]]] = None,
) -> pd.DataFrame:
    """
    Builds a season-long activity table with:
        kind: 'trade' | 'waiver'
        week: int
        ts: datetime (UTC)
        data: structured payload for HTML
    """
    season_weeks = list(range(1, 19))

    roster_name, roster_avatar = build_roster_display_maps(league_id)
    tx_by_week = get_transactions_by_week(league_id, season_weeks)

    rows: list[dict] = []
    players_map = players_map or {}
    pmap_get = players_map.get

    def pinfo(pid: str) -> dict[str, Any]:
        p = pmap_get(str(pid)) or {}
        gp = p.get
        return {
            "name": gp("name", str(pid)),
            "pos": gp("pos", ""),
            "team": gp("team", "FA"),
            "age": gp("age", None),
        }

    for week in season_weeks:
        txs = tx_by_week.get(week, []) or []

        for t in txs:
            ttype = t.get("type")
            ts_val = t.get("status_updated") or t.get("created") or 0
            ts = datetime.fromtimestamp(ts_val / 1000.0, tz=timezone.utc)

            if ttype in ("waiver", "waiver_add") and isinstance(t.get("adds"), dict):
                adds = t["adds"]
                by_rid: dict[str, list[dict]] = defaultdict(list)
                for pid, rid in adds.items():
                    by_rid[str(rid)].append(pinfo(pid))
                for rid, players in by_rid.items():
                    rows.append(
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

            if ttype == "trade":
                adds = t.get("adds") or {}
                drops = t.get("drops") or {}
                draft_picks = t.get("draft_picks") or []

                team_ids = sorted(
                    set(map(str, (t.get("roster_ids") or [])))  # base rosters in Sleeper tx
                    | {str(v) for v in adds.values()}  # teams receiving players
                    | {str(v) for v in drops.values()}  # teams sending players
                )

                team_objs = []
                for rid in team_ids:
                    gets = [pinfo(pid) for pid, to_rid in adds.items() if str(to_rid) == rid]
                    sends = [pinfo(pid) for pid, from_rid in drops.items() if str(from_rid) == rid]

                    try:
                        rid_int = int(rid)
                    except Exception:
                        rid_int = None

                    team_objs.append(
                        {
                            "rid": rid,
                            "roster_id": rid_int,
                            "name": roster_name.get(rid, f"Roster {rid}"),
                            "avatar": roster_avatar.get(rid),
                            "gets": gets,
                            "sends": sends,
                        }
                    )

                rows.append(
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
                continue

    df = pd.DataFrame(rows)
    if not df.empty:
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
            a = L.get("roster_id") or L.get("username") or L.get("name")
            b = R.get("roster_id") or R.get("username") or R.get("name")
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


def to_index(series):
    mu = series.mean()
    sigma = series.std(ddof=0)
    if sigma == 0 or pd.isna(sigma):
        return pd.Series(100.0, index=series.index)
    return 100 + 10 * (series - mu) / sigma


def build_team_strength(team_stats: pd.DataFrame) -> dict[str, float]:
    if "PowerScore" in team_stats.columns:
        base = team_stats["PowerScore"].astype(float)
    elif "Win%" in team_stats.columns:
        base = team_stats["Win%"].astype(float)
    elif "AVG" in team_stats.columns:
        base = team_stats["AVG"].astype(float)
    else:
        base = pd.Series(1.0, index=team_stats.index)

    min_v = float(base.min())
    max_v = float(base.max())
    if max_v == min_v:
        norm = pd.Series(0.5, index=base.index)
    else:
        norm = (base - min_v) / (max_v - min_v)

    strength_by_owner: dict[str, float] = {}
    for idx, row in team_stats.reset_index(drop=True).iterrows():
        owner = row.get("owner")
        if owner is None:
            continue
        strength_by_owner[owner] = float(norm.iloc[idx])

    return strength_by_owner


def compute_sos_by_team(
    all_matchups: Dict[int, List[dict]],
    team_strength: Dict[int, float],
    weeks_past: int,
    users: Dict[int, str],
) -> Dict[int, dict]:
    out: dict[Any, dict[str, Any]] = {
        owner: {"past_sos": 0.0, "past_cnt": 0, "ros_sos": 0.0, "ros_cnt": 0}
        for owner in team_strength
    }

    def _resolve_name(name: str) -> str:
        match = next(
            (
                u.get("metadata", {}).get("team_name") or name
                for u in users
                if u.get("display_name") == name
            ),
            name,
        )
        return match

    for w in range(1, weeks_past):
        for a, b in compute_week_opponents(all_matchups.get(w, [])):
            username = _resolve_name(a)
            username2 = _resolve_name(b)

            if username not in out or username2 not in out:
                continue

            out[username]["past_sos"] += team_strength[username2]
            out[username]["past_cnt"] += 1

            out[username2]["past_sos"] += team_strength[username]
            out[username2]["past_cnt"] += 1

    for w in range(weeks_past, 15):
        for a, b in compute_week_opponents(all_matchups.get(w, [])):
            username = _resolve_name(a)
            username2 = _resolve_name(b)

            if username not in out or username2 not in out:
                continue

            out[username]["ros_sos"] += team_strength[username2]
            out[username]["ros_cnt"] += 1

            out[username2]["ros_sos"] += team_strength[username]
            out[username2]["ros_cnt"] += 1

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

    def _indexify(values: List[float]) -> Tuple[float, float]:
        if not values:
            return 0.0, 0.0
        mu = sum(values) / len(values)
        var = sum((v - mu) ** 2 for v in values) / len(values)
        sigma = var ** 0.5
        return mu, sigma

    mu_p, sigma_p = _indexify(past_vals)
    mu_r, sigma_r = _indexify(ros_vals)

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

    def resolve_slot(match, side_key):
        rid = match.get(side_key)
        from_spec = match.get(f"{side_key}_from")

        if rid is not None:
            key = _k(rid)
            return {
                "label": roster_name.get(key, f"Roster {key}"),
                "avatar": roster_avatar_map.get(roster_name.get(key, "")),
                "kind": "team",
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

        return (
            f"<div class='{cls}'>"
            f"  <div class='team-main'>"
            f"    {img}"
            f"    <div class='team-text'><div class='team-name'>{slot['label']}</div></div>"
            f"  </div>"
            f"  <div class='team-score'>{score_text}</div>"
            f"</div>"
        )

    html_rounds = []
    for r in round_nums:
        round_label = {1: "Round 1", 2: "Semifinals", 3: "Finals 🏆"}.get(r, f"Round {r}")
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

    return "<div class='bracket'>" + "".join(html_rounds) + "</div>"


def seed_top6_from_team_stats(team_stats, roster_map):
    required_cols = {"owner", "Wins", "PF", "PA"}
    missing = required_cols - set(team_stats.columns)
    if missing:
        raise ValueError(f"team_stats missing required columns: {missing}")
    if not isinstance(roster_map, dict) or not roster_map:
        raise ValueError("roster_map must be a non-empty dict of {roster_id: team_name}")

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
        if seed > 6:
            break

    return seed_map


def render_standings_table(team_stats, length):
    rows = []

    df = team_stats.copy()
    df["WinPct"] = df["Win%"].astype(float)
    df = (
        df.sort_values(
            by=["Wins", "PF", "PA"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )

    df["Rank"] = df.index + 1

    for _, row in df.iterrows():
        record = f"{int(row['Wins'])}-{int(row['Losses'])}"
        if int(row.get("Ties", 0)):
            record += f"-{int(row['Ties'])}"

        streak = row.get("Streak", "")
        avatar = row.get("avatar", "")

        if avatar:
            img = (
                f"<img class='avatar sm' src='{avatar}' "
                "onerror=\"this.style.display='none'\">"
            )
        else:
            img = ""

        rows.append(
            f"""
            <tr>
              <td class="num">{int(row['Rank'])}</td>
              <td class="team">{img} {row['owner']}</td>
              <td>{record}</td>
              <td>{row['PF']:.1f}</td>
              <td>{row['PA']:.1f}</td>
              <td>{streak}</td>
              <td>{row['past_sos']:.1f}</td>
              <td>{row['ros_sos']:.1f}</td>
            </tr>
        """
        )
    total_rows = rows[:length] if len(rows) != length else rows
    return f"""
        <table class="standings-table">
          <h2>Standings</h2>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Team</th>
              <th>Record</th>
              <th>PF</th>
              <th>PA</th>
              <th>Streak</th>
              <th>SOS Past</th>
              <th>SOS Future</th>
            </tr>
          </thead>
          <tbody>
            {''.join(total_rows)}
          </tbody>
        </table>
    """


def render_teams_sidebar(teams: List[dict]) -> str:
    if not teams:
        return ""

    pill_buttons = []
    for idx, t in enumerate(teams):
        active_class = " active" if idx == 0 else ""
        label = t.get("username") or t["name"]
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
                    if extra_class:
                        row_cls += f" {extra_class}"
                    pos = p.get("pos")
                    pos_badge = f"<span class='pos-badge {pos}'>{pos}</span>" if pos else ""
                    nfl = p.get("nfl")
                    nfl_html = f"<span class='meta'>{nfl}</span>" if nfl else ""
                    out.append(
                        f"<div class='{row_cls}'>"
                        f"{pos_badge}"
                        f"<span class='pname'>{p['name']}</span>"
                        f"{nfl_html}"
                        "</div>"
                    )
            else:
                out.append("<div class='player-row empty'>None</div>")
            out.append("</div></div>")
            return "".join(out)

        sections = [
            render_player_list("Starters", t["starters"]),
            render_player_list("Bench", t["bench"]),
            render_player_list("Taxi", t["taxi"], extra_class="taxi"),
        ]

        picks = t.get("picks") or []
        picks_out: list[str] = []
        picks_out.append("<div class='team-section'>")
        picks_out.append("<div class='team-section-title'>Picks</div>")
        picks_out.append("<div class='player-list picks-list'>")
        if picks:
            for pk in picks:
                season = pk.get("season", "")
                rnd = pk.get("round", "")
                via = pk.get("original_owner")
                via_txt = f" (via {via})" if via else ""
                picks_out.append(
                    f"<div class='pick-row'>{season} • Round {rnd}{via_txt}</div>"
                )
        else:
            picks_out.append("<div class='player-row empty'>No picks</div>")
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


def build_picks_by_roster(
    num_future_seasons: int = 3,
    league: dict = None,
    rosters: List[dict] = None,
    traded: List[dict] = None,
) -> Dict[str, List[dict]]:
    current_season = int(league["season"])
    num_rounds = int(league["settings"].get("draft_rounds", 4))

    all_picks: List[dict] = []
    roster_ids = [int(r["roster_id"]) for r in rosters]

    for offset in range(num_future_seasons):
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
        return round(age, 1)
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


def _parse_stat_lines_for_pos(lines, pos_code: str) -> Dict[str, Any]:
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


def _parse_position_table_for_week(
    table,
    pos_code: str,
    target_week: int,
) -> Dict[str, Dict[str, Any]]:
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
        stats = _parse_stat_lines_for_pos(lines, pos_code)
        if stats:
            pos_players[player_name] = stats

    return pos_players


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

        pos_players = _parse_position_table_for_week(table, pos_code, target_week)
        result[pos_code] = pos_players

    return result


def build_and_save_week_stats_for_league(
    teams_index: Dict[str, Dict[str, Any]],
    season: int,
    week: int,
) -> Path:
    league_week_stats: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    print(f"[week_stats] building stats for the week")
    for team_abv in teams_index.keys():
        try:
            html = fetch_team_game_logs_html(team_abv, season)
            pos_player_stats = parse_team_week_pos_player_stats(html, week)
            league_week_stats[team_abv] = pos_player_stats
        except Exception as e:
            print(f"[week_stats] Error for {team_abv} week {week}: {e}")
            league_week_stats[team_abv] = {}

    out_path = path_week_stats(season, week)
    write_json(out_path, league_week_stats)
    print(f"[week_stats] Wrote → {out_path}")
    return out_path
