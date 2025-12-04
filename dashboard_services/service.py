from __future__ import annotations

import json
import numpy as np
import pandas as pd
import requests
import time
from collections import defaultdict
from datetime import datetime, timezone, date
from pathlib import Path
from plotly.offline import plot as plotly_plot
from typing import Dict, Any, Iterable, Tuple, Optional, List, Union, Callable
from zoneinfo import ZoneInfo

from dashboard_services.api import get_matchups, _avatar_url, get_nfl_state, avatar_from_users, get_transactions
from dashboard_services.matchups import build_matchup_preview
from dashboard_services.players import build_roster_display_maps
from dashboard_services.styles import recap_css, tickerCss
from dashboard_services.utils import safe_owner_name


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

    # Duplicate items for seamless loop
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


def matchup_cards_last_week(league_id: str,
                            df_weekly: pd.DataFrame,
                            roster_map: dict,
                            players_map: dict,
                            rosters: list,
                            users: list) -> tuple[int, str, dict]:
    """
    Returns: (week_number, html_for_matchup_cards, top_by_pos_dict)
      top_by_pos_dict: {'QB': [ {name, pts, nfl, team, owner}, ... up to 3 ], ...}
    """
    last_week = int(df_weekly["week"].max())
    raw = get_matchups(league_id, last_week) or []

    # group rows per matchup_id
    by_mid = defaultdict(list)
    for r in raw:
        by_mid[r.get("matchup_id")].append(r)

    # collect Top-3 per position from starters
    buckets = defaultdict(list)  # pos -> list of dict rows

    def pmeta(pid: str):
        p = players_map.get(str(pid), {})
        name = p.get("name") or str(pid)
        nfl = p.get("team") or "FA"
        pos = p.get("pos") or (p.get("fantasy_positions") or [""])[0]
        # Team DEF often comes through like "KC"
        if pid.isalpha() and 2 <= len(pid) <= 3 and pos in ("", None):
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
        username = next(
            (
                u.get("display_name")
                for u in users
                if u["user_id"] == next(
                (r["owner_id"] for r in rosters if str(r["roster_id"]) == str(ridL)),
                None
            )
            ),
            None
        )
        username2 = next(
            (
                u.get("display_name")
                for u in users
                if u["user_id"] == next(
                (r["owner_id"] for r in rosters if str(r["roster_id"]) == str(ridR)),
                None
            )
            ),
            None
        )

        # Owners / names / scores
        ln = safe_owner_name(roster_map, L.get("roster_id"))
        rn = safe_owner_name(roster_map, R.get("roster_id"))
        lp = float(L.get("points") or 0.0)
        rp = float(R.get("points") or 0.0)
        avatar = avatar_from_users(users, get_owner_id(rosters, ridL))
        avatar2 = avatar_from_users(users, get_owner_id(rosters, ridR))

        recordL = (0, 0)
        winsL, lossesL = recordL
        for r in rosters:
            if str(r.get("roster_id")) == ridL:
                recordL = (
                    r.get("settings", {}).get("wins", 0),
                    r.get("settings", {}).get("losses", 0)
                )
                winsL, lossesL = recordL
        recordR = (0, 0)
        winsR, lossesR = recordR
        for r in rosters:
            if str(r.get("roster_id")) == ridR:
                recordR = (
                    r.get("settings", {}).get("wins", 0),
                    r.get("settings", {}).get("losses", 0)
                )
                winsR, lossesR = recordR

        # Starters → per-player points + pos for Top-3 pools
        def harvest(row, owner_name):
            starters = [s for s in (row.get("starters") or []) if s]
            spts = row.get("starters_points") or []
            for i, pid in enumerate(starters):
                pid = str(pid)
                pts = float(spts[i]) if i < len(spts) and spts[i] is not None else float(
                    (row.get("players_points") or {}).get(pid, 0.0))
                name, nfl, pos = pmeta(pid)
                if pos:
                    buckets[pos].append({
                        "name": name, "pts": pts, "nfl": nfl, "owner": owner_name, "pid": pid
                    })

        harvest(L, ln)
        if R: harvest(R, rn)

        # winner highlight
        l_cls = "win" if lp > rp else "loss" if rp > lp else "tie"
        r_cls = "win" if rp > lp else "loss" if lp > rp else "tie"

        cards.append(f"""
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
        """)

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
        if pid in r.get("players", []):
            rid = str(r["roster_id"])
            return roster_map.get(rid, f"Roster {rid}")

    return "Free Agent"


def render_top_three(top_by_pos: dict, rosters, roster_map) -> str:
    def card(pos, rows):
        if not rows:
            return f"<div class='side-card'><h3>{pos}</h3><div class='muted'>No data</div></div>"
        lis = []
        for i, r in enumerate(rows, start=1):
            team = r.get("nfl") or r.get("team", "")
            pts = r.get("pts") or r.get("points")
            if not r.get("owner_id"):
                owner = fantasy_team_for_player(r['pid'], rosters, roster_map)
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

    # Wrap all cards inside a responsive 2-column grid
    blocks = [card(pos, top_by_pos.get(pos, [])) for pos in ["QB", "RB", "WR", "TE", "K", "DEF"]]
    return "<div class='sidebar-grid'>" + "".join(blocks) + "</div>"


def render_week_recap_tab(league_id: str,
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

    # Match the same “card + inner markup + (optional) script” structure as the carousel function
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

    # Build user lookup (metadata avatar → team logo)
    user_by_id = {u["user_id"]: u for u in users}

    # Create fallback display names
    user_fallback = {
        u["user_id"]: (
                (u.get("metadata") or {}).get("team_name")
                or u.get("display_name")
                or u.get("username")
                or str(u["user_id"])
        )
        for u in users
    }

    # ---- Roster ID → Display Name (string keys) ----
    roster_map: dict[str, str] = {}
    for r in rosters:
        rid = str(r["roster_id"])
        owner_id = r.get("owner_id")
        display = user_fallback.get(owner_id, f"Roster {rid}")
        roster_map[rid] = display

    # Pre-build matchups_by_week for SOS later
    matchups_by_week = build_matchups_by_week(
        league_id, range(1, 18), roster_map, players
    )

    # ---- Owner → Avatar URL ----
    owner_avatar: dict[str, Union[str, None]] = {}
    for r in rosters:
        rid = str(r["roster_id"])
        owner_id = r.get("owner_id")
        display = roster_map.get(rid, f"Roster {rid}")

        avatar_id = None
        if owner_id in user_by_id:
            user_data = user_by_id[owner_id]
            user_meta = user_data.get("metadata") or {}
            # try to get Sleeper avatar id from users list
            u_id = next(
                (u.get("avatar") for u in users if u["user_id"] == owner_id),
                None,
            )
            avatar_id = (
                    user_meta.get("avatar")  # team logo if set
                    or (f"https://sleepercdn.com/avatars/{u_id}" if u_id else None)
            )

        owner_avatar[display] = _avatar_url(avatar_id)

    # ---- Weekly matchup data ----
    weekly_rows = []
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

    # ---- Points against pairing ----
    df_weekly["points_against"] = np.nan
    for (_, _mid), grp in df_weekly.groupby(["week", "matchup_id"]):
        if len(grp) == 2:
            i1, i2 = grp.index.tolist()
            p1, p2 = df_weekly.loc[i1, "points"], df_weekly.loc[i2, "points"]
            df_weekly.loc[i1, "points_against"] = p2
            df_weekly.loc[i2, "points_against"] = p1

    # Attach team avatars (logos) to weekly rows
    df_weekly["avatar"] = df_weekly["owner"].map(owner_avatar)

    # ---- Finalized flag ----
    _state = get_nfl_state()
    current_leg = int(_state.get("leg") or _state.get("week") or 0)
    df_weekly["finalized"] = df_weekly["week"] < current_leg

    finalized_mask = df_weekly["finalized"] == True
    df_finalized = df_weekly[finalized_mask].copy()

    # ---- Aggregate stats ----
    records = _compute_team_records(df_finalized.copy())
    team_stats = _aggregate_team_stats(df_finalized.copy(), records)

    # merge avatar into team_stats (by owner name)
    team_stats = team_stats.merge(
        pd.Series(owner_avatar, name="avatar"),
        left_on="owner",
        right_index=True,
        how="left",
    )

    # Last 3 games avg
    last3 = (
        df_finalized.sort_values(["owner", "week"])
        .groupby("owner")["points"]
        .apply(lambda s: s.tail(3).mean() if len(s) else 0.0)
        .rename("Last3")
        .reset_index()
    )
    team_stats = team_stats.merge(last3, on="owner", how="left")
    team_stats["Last3"] = team_stats["Last3"].fillna(0.0)

    # ---- Z-score helper ----
    def _z(series):
        s = pd.Series(series, dtype="float64")
        sd = float(s.std(ddof=0))
        if sd == 0 or np.isnan(sd):
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / sd

    # Win% (handle both explicit Win% or Wins/Losses/Ties)
    if "Win%" in team_stats.columns:
        win_pct = team_stats["Win%"].fillna(0.0)
    else:
        if "Ties" in team_stats.columns:
            ties = team_stats["Ties"].fillna(0.0)
        else:
            ties = 0.0
        win_pct = (
                (team_stats["Wins"] + 0.5 * ties)
                / team_stats["G"].replace(0, np.nan)
        ).fillna(0.0)

    # Input series for power score components
    avg_pts = team_stats.get("AVG", pd.Series(0.0, index=team_stats.index)).fillna(0.0)
    cons_inv = -team_stats.get("STD", pd.Series(team_stats["STD"].mean(), index=team_stats.index)).fillna(
        team_stats["STD"].mean()
    )
    ceiling = team_stats.get("MAX", pd.Series(0.0, index=team_stats.index)).fillna(0.0)
    last3_series = team_stats["Last3"].fillna(0.0)

    # Z-scores
    team_stats["Z_WinPercentage"] = _z(win_pct)
    team_stats["Z_Avg"] = _z(avg_pts)
    team_stats["Z_Last3"] = _z(last3_series)
    team_stats["Z_Consistency"] = _z(cons_inv)
    team_stats["Z_Ceiling"] = _z(ceiling)

    # Power score
    W_WIN, W_AVG, W_LAST3, W_CONS, W_CEIL = 0.2, 0.3, 0.15, 0.20, 0.15
    team_stats["Win%"] = win_pct
    team_stats["PowerScore"] = (
            W_WIN * team_stats["Z_WinPercentage"]
            + W_AVG * team_stats["Z_Avg"]
            + W_LAST3 * team_stats["Z_Last3"]
            + W_CONS * team_stats["Z_Consistency"]
            + W_CEIL * team_stats["Z_Ceiling"]
    )

    # ---- Strength of Schedule (SOS) ----
    sos = build_team_strength(team_stats)
    last_week = int(df_weekly["week"].max())
    sos_dict = compute_sos_by_team(matchups_by_week, sos, last_week, users)
    sos_df = (
        pd.DataFrame.from_dict(sos_dict, orient="index")
        .reset_index()
        .rename(columns={"index": "owner"})
    )
    team_stats = team_stats.merge(sos_df, on="owner", how="left")

    # ---- Streaks ----
    streaks_df = compute_streaks(df_finalized.copy())
    # just merge streaks_df directly; don't try to index with df_weekly mask
    team_stats = team_stats.merge(streaks_df, on="owner", how="left")

    # Fill empties
    team_stats["StreakType"] = team_stats["StreakType"].fillna("")
    team_stats["StreakLen"] = team_stats["StreakLen"].fillna(0).astype(int)
    team_stats["Streak"] = team_stats["Streak"].fillna("")

    return df_weekly, team_stats, roster_map


def build_league_summary(team_stats, df_weekly) -> dict:
    """
    team_stats: DataFrame with columns like ['owner', 'PF', 'PA', 'MAX', 'MIN', 'AVG', 'STD', ...]
    df_weekly:  DataFrame with ['week', 'owner', 'points', ...]
    """
    summary = {}

    if not team_stats.empty:
        best_pf_row = team_stats.loc[team_stats["PF"].idxmax()]
        worst_pf_row = team_stats.loc[team_stats["PF"].idxmin()]
        best_avg_row = team_stats.loc[team_stats["AVG"].idxmax()]
        most_vol_row = team_stats.loc[team_stats["STD"].idxmax()]

        # Luck via PF - PA
        ts = team_stats.copy()
        ts["pf_minus_pa"] = ts["PF"] - ts["PA"]
        luckiest_row = ts.loc[ts["pf_minus_pa"].idxmax()]
        unluckiest_row = ts.loc[ts["pf_minus_pa"].idxmin()]

        summary["best_pf"] = {
            "owner": best_pf_row["owner"],
            "pf": float(best_pf_row["PF"]),
        }
        summary["worst_pf"] = {
            "owner": worst_pf_row["owner"],
            "pf": float(worst_pf_row["PF"]),
        }
        summary["best_avg"] = {
            "owner": best_avg_row["owner"],
            "avg": float(best_avg_row["AVG"]),
        }
        summary["most_vol"] = {
            "owner": most_vol_row["owner"],
            "std": float(most_vol_row["STD"]),
        }
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
    """Compute win/loss/tie records based on points scored."""
    wins = defaultdict(int)
    losses = defaultdict(int)
    ties = defaultdict(int)
    games_played = defaultdict(int)

    for (_, mid), group in df.groupby(["week", "matchup_id"]):
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
    for owner in sorted(set(df["owner"])):
        w = wins[owner];
        l = losses[owner];
        t = ties[owner];
        g = games_played[owner]
        results.append({
            "owner": owner,
            "Wins": w,
            "Losses": l,
            "Ties": t,
            "G": g,
            "Win%": (w + 0.5 * t) / g if g else 0.0
        })

    return pd.DataFrame(results)


def _aggregate_team_stats(df_weekly: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """Aggregate team statistics from weekly data and merge with records."""
    stats = df_weekly.groupby("owner").agg(
        PF=("points", "sum"),
        PA=("points_against", "sum"),
        AVG=("points", "mean"),
        MAX=("points", "max"),
        MIN=("points", "min"),
        STD=("points", "std")
    ).reset_index()

    team_stats = stats.merge(records, on="owner", how="left")

    team_stats["Record"] = team_stats[["Wins", "Losses", "Ties"]].apply(
        lambda r: f"{int(r.Wins)}-{int(r.Losses)}" +
                  (f"-{int(r.Ties)}" if r.Ties else ""),
        axis=1
    )

    return team_stats


def get_owner_id(
        rosters: Optional[list[dict]] = None,
        roster_id: Optional[str] = None
) -> Optional[str]:
    return next((r["owner_id"] for r in rosters if str(r.get("roster_id")) == str(roster_id)), None)


def build_matchups_by_week(league_id, weeks, roster_map, players_map):
    by_week = {}
    for w in weeks:
        # Your existing builder; pass None for projections
        matchups = build_matchup_preview(
            league_id=league_id,
            week=w,
            roster_map=roster_map,
            players_map=players_map,
        )
        by_week[w] = matchups or []
    return by_week


def _weekly_results_from_df(df_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Returns rows: owner | week | result where result in {'W','L','T'}.
    """
    rows = []
    for (_, mid), g in df_weekly.groupby(["week", "matchup_id"]):
        g = g.sort_values("roster_id")  # any stable order
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
    """
    Input is a list like ['W','L','W','W'] in chronological order.
    Returns (type, length) e.g. ('W', 2). If empty, ('', 0).
    """
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
    """
    Output: owner | StreakType ('W'/'L'/'T'/'') | StreakLen | Streak label (e.g., 'W3')
    """
    res = _weekly_results_from_df(df_weekly)
    if res.empty:
        return pd.DataFrame(columns=["owner", "StreakType", "StreakLen", "Streak"])

    out = []
    for owner, g in res.sort_values("week").groupby("owner"):
        typ, length = _current_streak(g["result"].tolist())
        label = f"{typ}{length}" if typ and length else ""
        out.append({"owner": owner, "StreakType": typ, "StreakLen": int(length), "Streak": label})
    return pd.DataFrame(out)


def get_transactions_by_week(league_id: str, season_weeks: list[int]) -> dict[int, list[dict]]:
    """
    Fetches transactions week-by-week from Sleeper:
        /league/{league_id}/transactions/{week}

    Returns:
        {
            1: [ {...}, {...} ],
            2: [ ... ],
            ...
        }
    """
    results: dict[int, list[dict]] = {}

    for w in season_weeks:
        try:
            tx = get_transactions(league_id=league_id, week=w)
            if isinstance(tx, list):
                results[w] = tx
            else:
                results[w] = []
        except Exception as e:
            print(f"[transactions] Week {w} failed → {e}")
            results[w] = []

    return results


def build_week_activity(
        league_id: str,
        season: int,
        players_map: Optional[Dict[str, Dict[str, str]]] = None,
        season_weeks: list[int] = None
) -> pd.DataFrame:
    """
    Builds a season-long activity table with:
        kind: 'trade' | 'waiver'
        week: int
        ts: datetime (UTC)
        data: structured payload for HTML
            - trade: {'teams': [...], 'draft_picks': [...]}
            - waiver: {'rid','name','avatar','adds':[players]}
    """

    if players_map is None:
        players_map = load_players_index()

    if season_weeks is None:
        season_weeks = list(range(1, 19))  # NFL always 1–18 since 2021

    # As before
    roster_name, roster_avatar = build_roster_display_maps(league_id)

    # 🔥 Fetch ALL weekly transactions grouped by week
    tx_by_week = get_transactions_by_week(league_id, season_weeks)

    rows = []

    def pinfo(pid: str) -> dict[str, str]:
        p = players_map.get(str(pid)) or {}
        return {
            "name": p.get("name", str(pid)),
            "pos": p.get("pos", ""),
            "team": p.get("team", "FA"),
            "age": p.get("age", None),
        }

    for week in season_weeks:
        txs = tx_by_week.get(week, []) or []

        for t in txs:
            ttype = t.get("type")
            ts = datetime.fromtimestamp(
                (t.get("status_updated") or t.get("created") or 0) / 1000.0,
                tz=timezone.utc
            )

            # --------------------------
            # WAIVERS
            # --------------------------
            if ttype in ("waiver", "waiver_add") and isinstance(t.get("adds"), dict):
                adds = t["adds"]  # {player_id: roster_id}
                by_rid: dict[str, list[dict]] = defaultdict(list)

                for pid, rid in adds.items():
                    by_rid[str(rid)].append(pinfo(pid))

                for rid, players in by_rid.items():
                    rows.append({
                        "kind": "waiver",
                        "week": week,
                        "ts": ts,
                        "data": {
                            "rid": rid,
                            "name": roster_name.get(rid, f"Roster {rid}"),
                            "avatar": roster_avatar.get(rid),
                            "adds": players,
                        }
                    })
                continue

            # --------------------------
            # TRADES
            # --------------------------
            if ttype == "trade":
                adds = t.get("adds") or {}  # player_id → new_owner_rid
                drops = t.get("drops") or {}  # player_id → old_owner_rid
                draft_picks = t.get("draft_picks") or []

                # Collect roster IDs involved
                team_ids = sorted(
                    set(
                        list(map(str, (t.get("roster_ids") or [])))
                        + list({str(v) for v in adds.values()})
                        + list({str(v) for v in drops.values()})
                    )
                )

                team_objs = []
                for rid in team_ids:
                    # Players received by this roster
                    gets = [pinfo(pid) for pid, to_rid in adds.items() if str(to_rid) == rid]

                    # Players sent by this roster
                    sends = [pinfo(pid) for pid, from_rid in drops.items() if str(from_rid) == rid]

                    # int form for draft pick matching
                    rid_int = None
                    try:
                        rid_int = int(rid)
                    except:
                        pass

                    team_objs.append({
                        "rid": rid,
                        "roster_id": rid_int,
                        "name": roster_name.get(rid, f"Roster {rid}"),
                        "avatar": roster_avatar.get(rid),
                        "gets": gets,
                        "sends": sends
                    })

                rows.append({
                    "kind": "trade",
                    "week": week,
                    "ts": ts,
                    "data": {
                        "teams": team_objs,
                        "draft_picks": draft_picks
                    }
                })
                continue

    # --------------------------
    # Build DataFrame
    # --------------------------
    df = pd.DataFrame(rows)

    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df = df.sort_values("ts", ascending=False).reset_index(drop=True)

    return df


def compute_week_opponents(matchups_week: Iterable[Dict[str, Any]]) -> List[Tuple[Any, Any]]:
    """
    Returns list of opponent pairs for the week.
    - Old shape: [{"matchup_id": int, "roster_id": int, ...}, ...]  -> pairs of roster_ids
    - New shape: [{"matchup_id": int, "left": {...}, "right": {...}}, ...] -> pairs of ids/usernames/names
    """
    # If a single dict was passed, normalize to list
    if isinstance(matchups_week, dict):
        matchups_week = [matchups_week]

    pairs: List[Tuple[Any, Any]] = []

    # Detect new shape quickly
    new_shape = any(("left" in m and "right" in m) for m in matchups_week)

    if new_shape:
        for m in matchups_week:
            if "left" not in m or "right" not in m:
                continue
            L = m["left"] or {}
            R = m["right"] or {}
            a = L.get("roster_id") or L.get("username") or L.get("name")
            b = R.get("roster_id") or R.get("username") or R.get("name")
            if a is not None and b is not None:
                pairs.append((a, b))
        return pairs

    # Legacy shape: group by matchup_id, pair roster_ids
    by_id: Dict[Any, List[Any]] = {}
    for m in matchups_week:
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
    """
    Build a simple "team strength" metric per owner for SOS calculations.

    Uses:
      - PowerScore if available
      - else WinPct if available
      - else AVG (points per game)
    Normalizes everything to 0–1.
    Returns: { owner_name: strength_float }
    """

    if "PowerScore" in team_stats.columns:
        base = team_stats["PowerScore"].astype(float)
    elif "Win%" in team_stats.columns:
        base = team_stats["Win%"].astype(float)
    elif "AVG" in team_stats.columns:
        base = team_stats["AVG"].astype(float)
    else:
        # fallback: everyone equal strength
        base = pd.Series(1.0, index=team_stats.index)

    # normalize to 0..1
    min_v = float(base.min())
    max_v = float(base.max())
    if max_v == min_v:
        norm = pd.Series(0.5, index=base.index)
    else:
        norm = (base - min_v) / (max_v - min_v)

    # map to owner names
    strength_by_owner: dict[str, float] = {}
    for idx, row in team_stats.reset_index(drop=True).iterrows():
        owner = row.get("owner")
        if owner is None:
            continue
        strength_by_owner[owner] = float(norm.iloc[idx])

    return strength_by_owner


def compute_sos_by_team(
        all_matchups: Dict[int, List[dict]],  # week -> list of rows
        team_strength: Dict[int, float],
        weeks_past: int,
        users: Dict[int, str],
) -> Dict[int, dict]:
    """
    Compute strength of schedule for each team.

    Uses team_strength as the underlying rating (e.g. PowerScore) and returns
    a normalized index:

        past_sos, ros_sos ~= 100 => league-average difficulty
        > 100 => harder schedule
        < 100 => easier schedule
    """
    # initialize
    out = {
        owner: {"past_sos": 0.0, "past_cnt": 0, "ros_sos": 0.0, "ros_cnt": 0}
        for owner in team_strength
    }

    # helper to resolve display_name -> team_name (username)
    def _resolve_name(name: str) -> str:
        # try to map Sleeper display_name to metadata.team_name
        match = next(
            (
                u.get("metadata", {}).get("team_name") or name
                for u in users
                if u.get("display_name") == name
            ),
            name,
        )
        return match

    # --- Past SOS: weeks 1 .. weeks_past-1 ---
    for w in range(1, weeks_past):
        for a, b in compute_week_opponents(all_matchups.get(w, [])):
            username = _resolve_name(a)
            username2 = _resolve_name(b)

            # skip if not in team_strength (safety)
            if username not in out or username2 not in out:
                continue

            out[username]["past_sos"] += team_strength[username2]
            out[username]["past_cnt"] += 1

            out[username2]["past_sos"] += team_strength[username]
            out[username2]["past_cnt"] += 1

    # --- Future (ROS) SOS: weeks weeks_past .. 14 (or 15 exclusive) ---
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

    # --- Step 1: convert sums to averages (raw SOS) ---
    past_vals = []
    ros_vals = []
    for team, v in out.items():
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

    # --- Step 2: normalize to index (100 = avg, 10 points = 1 std dev) ---

    def _indexify(values: List[float]) -> Tuple[float, float]:
        if not values:
            return 0.0, 0.0
        mu = sum(values) / len(values)
        var = sum((v - mu) ** 2 for v in values) / len(values)
        sigma = var ** 0.5
        return mu, sigma

    mu_p, sigma_p = _indexify(past_vals)
    mu_r, sigma_r = _indexify(ros_vals)

    for team, v in out.items():
        # past SOS index
        if v["past_cnt"] and sigma_p > 0:
            v["past_sos"] = 100.0 + 10.0 * (v["past_sos"] - mu_p) / sigma_p
        elif v["past_cnt"]:
            # everyone had identical past SOS -> flat 100
            v["past_sos"] = 100.0
        else:
            # no past games: keep 0.0 as "N/A"
            v["past_sos"] = 0.0

        # future SOS index
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
        seed_map=None,  # NEW: { roster_id(str/int) -> seed(int) }
):
    """
    Render an HTML playoff bracket from Sleeper's /winners_bracket endpoint.

    seed_map (optional):
        Dict mapping roster_id -> seed (1 = best, 2 = next, etc.).
        When present, BYE teams and ordering are based on seed instead of roster id.
    """
    if not winners_bracket:
        return "<div class='po-empty'>No playoff bracket available.</div>"

    match_scores = match_scores or {}
    seed_map = seed_map or {}

    # normalize keys to strings
    def _k(x):
        return str(x) if x is not None else None

    # how to sort a roster id: by seed first, then by roster id as tiebreaker
    def seed_key(rid):
        if rid is None:
            return 9999
        # allow both str and int keys in seed_map
        s = seed_map.get(str(rid))
        if s is None:
            s = seed_map.get(int(rid), None) if isinstance(rid, (int, str)) and str(rid).isdigit() else None
        # if no seed, use a large-ish value plus roster_id for stable ordering
        try:
            rid_int = int(rid)
        except Exception:
            rid_int = 9999
        return (s if s is not None else 9999, rid_int)

    # normalize maps (roster_id -> name/avatar)
    roster_name = {_k(k): v for k, v in (roster_name_map or {}).items()}

    # --- 1) Find bye teams (in bracket but not in Round 1) ---
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
    # sort by SEED, not roster id
    bye_rids_sorted = sorted(bye_rids, key=seed_key)

    # --- 2) Add synthetic Round 1 BYE matches, top + bottom ---
    extended_bracket = list(winners_bracket)
    round1_override = None  # will hold explicit order if we build byes

    if bye_rids_sorted:
        existing_ids = [m.get("m") for m in winners_bracket if isinstance(m.get("m"), int)]
        next_m = max(existing_ids) + 1 if existing_ids else 1

        # create bye matches in sorted (by seed) order
        bye_matches = []
        for rid in bye_rids_sorted:
            bye_matches.append({
                "m": next_m,
                "r": 1,
                "w": None,
                "l": None,
                "t1": rid,  # real team (seeded)
                "t2": None,  # bye side
                "t1_from": None,
                "t2_from": None,
                "is_bye": True,
            })
            next_m += 1

        # existing non-bye Round 1 matches
        r1_existing = [m for m in extended_bracket if m.get("r") == 1 and not m.get("is_bye")]
        non_r1 = [m for m in extended_bracket if m.get("r") != 1]

        if len(bye_matches) == 1:
            # single bye at the top
            new_r1 = bye_matches[:1] + r1_existing
        elif len(bye_matches) >= 2:
            # first bye at top, last bye at bottom, any others in the middle
            middle_byes = bye_matches[1:-1]
            new_r1 = [bye_matches[0]] + r1_existing + middle_byes + [bye_matches[-1]]
        else:
            new_r1 = r1_existing

        extended_bracket = non_r1 + new_r1
        round1_override = new_r1  # remember explicit order for round 1

    winners_bracket = extended_bracket

    # index by match id
    match_by_id = {m["m"]: m for m in winners_bracket if "m" in m}

    # group by round
    rounds = defaultdict(list)
    for m in winners_bracket:
        r = m.get("r")
        if r is None:
            continue
        rounds[r].append(m)

    if not rounds:
        return "<div class='po-empty'>No playoff bracket available.</div>"

    # override round 1 order if we built a custom one
    if round1_override:
        rounds[1] = round1_override

    round_nums = sorted(rounds.keys())
    for r in round_nums:
        if r == 1 and round1_override:
            continue
        rounds[r].sort(key=lambda x: x.get("m", 0))

    # ---- rest of your existing resolve/render code unchanged ----
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
            src_type, src_mid = next(iter(from_spec.items()))  # ("w",1) or ("l",2)

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

        img = (
            f"<div class='team-avatar'><img src='{slot['avatar']}' "
            "onerror=\"this.style.display='none'\"></div>"
            if slot.get("avatar")
            else "<div class='team-avatar'></div>"
        )

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
    """
    Produce a seed map { roster_id_str: seed } for the top 6 teams.

    Uses team_stats columns:
      - 'owner' (team name, matching roster_map values)
      - 'wins'
      - 'PF'
      - 'PA'

    Tie-breakers:
      1) wins (desc)
      2) PF (desc)
      3) PA (asc)
      4) owner name (asc; just to keep things stable)
    """

    # ---- 0) Basic validation ----
    required_cols = {"owner", "Wins", "PF", "PA"}
    missing = required_cols - set(team_stats.columns)
    if missing:
        raise ValueError(f"team_stats missing required columns: {missing}")

    if not isinstance(roster_map, dict) or not roster_map:
        raise ValueError("roster_map must be a non-empty dict of {roster_id: team_name}")

    # ---- 1) Build reverse map: team_name -> roster_id ----
    # Normalize names a bit (strip whitespace)
    name_to_rid = {}
    for rid, name in roster_map.items():
        if not isinstance(name, str):
            continue
        key = name.strip()
        if key:
            name_to_rid[key] = str(rid).strip()

    # ---- 2) Sort team_stats by record + PF + PA ----
    df = team_stats.copy()
    df["owner_norm"] = df["owner"].astype(str).str.strip()

    df_sorted = (
        df.sort_values(
            by=["Wins", "PF", "PA", "owner_norm"],
            ascending=[False, False, True, True],  # wins desc, PF desc, PA asc
        )
        .reset_index(drop=True)
    )

    # ---- 3) Build seed map for top 6, using roster IDs from roster_map ----
    seed_map: dict[str, int] = {}
    seed = 1

    for _, row in df_sorted.iterrows():
        owner_name = row["owner_norm"]
        rid = name_to_rid.get(owner_name)

        # Skip if we can't map this owner to a roster_id
        if not rid:
            continue

        if rid in seed_map:
            # Already assigned a seed (shouldn't normally happen, but safe)
            continue

        seed_map[rid] = seed
        seed += 1

        if seed > 6:
            break

    return seed_map


def render_standings_table(team_stats, length):
    rows = []

    # Sort by Wins, Win%, PF, PA
    df = team_stats.copy()
    df["WinPct"] = df["Win%"].astype(float)
    df = (
        df.sort_values(
            by=["Wins", "PF", "PA"],
            ascending=[False, False, True],  # PA lower is better
        )
        .reset_index(drop=True)
    )

    # Add Rank column (1..N)
    df["Rank"] = df.index + 1

    for _, row in df.iterrows():
        record = f"{int(row['Wins'])}-{int(row['Losses'])}"
        if int(row.get("Ties", 0)):
            record += f"-{int(row['Ties'])}"

        streak = row.get("Streak", "")
        avatar = row.get("avatar", "")

        img = (
            f"<img class='avatar sm' src='{avatar}' "
            "onerror=\"this.style.display='none'\">"
            if avatar else ""
        )

        rows.append(f"""
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
        """)
    if len(rows) != length:
        total_rows = rows[:length]
    else:
        total_rows = rows
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

    # Tabs row
    pill_buttons = []
    for idx, t in enumerate(teams):
        active_class = " active" if idx == 0 else ""
        label = t.get("username") or t["name"]
        pill_buttons.append(
            f"<button class='manager-pill{active_class}' "
            f"data-team-id='{t['roster_id']}'>{label}</button>"
        )
    header_html = "<div class='manager-pills-carousel'><button class='pill-arrow pill-arrow-left' type='button'>&lsaquo;</button> <div class='manager-pills-row'>" + "".join(
        pill_buttons) + "</div><button class='pill-arrow pill-arrow-right' type='button'>&rsaquo;</button></div>"

    # panels: one per team, only first visible
    panel_html_parts = []
    for idx, t in enumerate(teams):
        active_class = " active" if idx == 0 else ""

        def render_player_list(title: str, players: List[dict], extra_class: str = "") -> str:
            out = []
            out.append("<div class='team-section'>")
            out.append(f"<div class='team-section-title'>{title}</div>")
            out.append("<div class='player-list'>")
            if players:
                for p in players:
                    row_cls = "player-row"
                    if extra_class:
                        row_cls += f" {extra_class}"
                    pos_badge = ""
                    if p.get("pos"):
                        pos_badge = f"<span class='pos-badge {p['pos']}'>{p['pos']}</span>"
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

        sections = []
        sections.append(render_player_list("Starters", t["starters"]))
        sections.append(render_player_list("Bench", t["bench"]))
        sections.append(render_player_list("Taxi", t["taxi"], extra_class="taxi"))

        # picks at the bottom
        picks = t.get("picks") or []
        picks_out = []
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

    # whole sidebar card
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
    """
    Returns:
      {
        '1': [ {season, round, original_owner}, ... ],
        '2': [ ... ],
        ...
      }

    - Keys are *current owner* roster_ids (as strings)
    - `original_owner` is the roster_id that started with the pick
    """  # can be empty list

    current_season = int(league["season"])
    num_rounds = int(league["settings"].get("draft_rounds", 4))

    # Base pick pool: everyone owns their own picks for each season/round
    # We store picks as dicts we can mutate when applying trades.
    all_picks: List[dict] = []

    roster_ids = [int(r["roster_id"]) for r in rosters]

    for offset in range(num_future_seasons):
        season = current_season + offset
        for rid in roster_ids:
            for rnd in range(1, num_rounds + 1):
                all_picks.append({
                    "season": season,
                    "round": rnd,
                    "original_roster_id": rid,
                    "owner_roster_id": rid,  # will change when trades applied
                })

    # Apply traded picks
    # Sleeper's traded_picks are for "future" picks, not past drafts
    for tp in traded:
        try:
            season = int(tp["season"])
            rnd = int(tp["round"])
            original = int(tp["roster_id"])  # original owner roster_id
            new_owner = int(tp["owner_id"])  # CURRENT owner roster_id
        except (KeyError, ValueError, TypeError):
            continue

        # Find the matching pick in all_picks and update its owner
        for p in all_picks:
            if (
                    p["season"] == season and
                    p["round"] == rnd and
                    p["original_roster_id"] == original
            ):
                p["owner_roster_id"] = new_owner

    # Group by current owner roster_id
    picks_by_roster: Dict[str, List[dict]] = {}

    for p in all_picks:
        owner_key = str(p["owner_roster_id"])
        picks_by_roster.setdefault(owner_key, []).append({
            "season": p["season"],
            "round": p["round"],
            "original_owner": str(p["original_roster_id"]),
        })

    # Optional: sort picks by season, round
    for rid in picks_by_roster:
        picks_by_roster[rid].sort(key=lambda x: (x["season"], x["round"]))

    return picks_by_roster


def age_from_bday(bday: Optional[str]) -> Optional[float]:
    """
    Compute decimal age in years as of Sept 1 of `season`.

    Returns values like:
      22.4, 26.7, 30.1, etc.
    """
    if not bday:
        return None

    try:
        # Handle formats like "1999-05-14" or "1999-05-14T00:00:00Z"
        parts = bday.split("T")[0].split("/")
        month, day, year = map(int, parts[:3])
        dob = date(year, month, day)

        # Dynasty age baseline: as of Sept 1 of the current season
        as_of = date.today()

        # Convert to decimal years
        days = (as_of - dob).days
        age = days / 365.25

        # Round to one decimal place (26.7 style)
        return round(age, 1)

    except Exception:
        return None


def pill(s):
    return f"<span class='badge'>{s}</span>"


def build_standings_map(team_stats, roster_map) -> dict[int, int]:
    """
    Returns {roster_id: seed}, where seed 1 is best.
    Adjust sort keys as needed (wins, PF, etc.).
    """
    # Example: sort by wins desc, then PF desc
    ordered = (
        team_stats
        .sort_values(["Wins", "PF"], ascending=[False, False])
        .reset_index(drop=True)
    )
    owner_to_rid = {owner: rid for rid, owner in roster_map.items()}

    standings = {}
    for idx, row in ordered.iterrows():
        # adjust column name if it's different (e.g. 'roster_id')
        owner = row["owner"]
        rid = owner_to_rid.get(owner)
        seed = idx + 1  # 1-based
        standings[rid] = seed

    return standings
