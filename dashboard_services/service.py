import numpy as np
import pandas as pd
import requests
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Any, Iterable, Tuple, Optional, List, Union, Callable

from .api import get_matchups, get_users, get_rosters, _avatar_url, get_nfl_state, _avatar_from_users, fetch_json
from .matchups import build_matchup_preview
from .players import build_roster_display_maps, get_players_map
from .styles import recap_css, tickerCss
from .utils import safe_owner_name


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


def _matchup_cards_last_week(league_id: str,
                             df_weekly: pd.DataFrame,
                             roster_map: dict,
                             players_map: dict) -> tuple[int, str, dict]:
    """
    Returns: (week_number, html_for_matchup_cards, top_by_pos_dict)
      top_by_pos_dict: {'QB': [ {name, pts, nfl, team, owner}, ... up to 3 ], ...}
    """
    last_week = int(df_weekly["week"].max())
    raw = get_matchups(league_id, last_week) or []
    users = get_users(league_id)
    rosters = get_rosters(league_id)

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
        avatar = _avatar_from_users(users, get_owner_id(rosters, ridL))
        avatar2 = _avatar_from_users(users, get_owner_id(rosters, ridR))

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

    # Top-3 by position (normalize/limit to common positions)
    want_positions = ["QB", "RB", "WR", "TE", "K", "DEF"]
    top_by_pos = {}
    for pos in want_positions:
        pool = sorted(buckets.get(pos, []), key=lambda x: x["pts"], reverse=True)[:3]
        top_by_pos[pos] = pool

    return last_week, "".join(cards), top_by_pos


def _render_top_three_sidebar(top_by_pos: dict) -> str:
    def card(pos, rows):
        if not rows:
            return f"<div class='side-card'><h3>{pos}</h3><div class='muted'>No data</div></div>"
        lis = []
        for i, r in enumerate(rows, start=1):
            place = "first" if i == 1 else "second" if i == 2 else "third"
            lis.append(
                f"<div class='side-row'>"
                f"  <span class='rank rank-{place}'>{i}</span>"
                f"  <div class='who'>"
                f"    <div class='name'>{r['name']}</div>"
                f"    <div class='sub'>{r['nfl']} • {r['owner']}</div>"
                f"  </div>"
                f"  <div class='pts'>{r['pts']:.1f}</div>"
                f"</div>"
            )
        return f"<div class='side-card'><h3>{pos}</h3>{''.join(lis)}</div>"

    # Wrap all cards inside a responsive 2-column grid
    blocks = [card(pos, top_by_pos.get(pos, [])) for pos in ["QB", "RB", "WR", "TE", "K", "DEF"]]
    return "<div class='sidebar-grid'>" + "".join(blocks) + "</div>"


def render_week_recap_tab(league_id: str,
                          df_weekly: pd.DataFrame,
                          roster_map: dict,
                          players_map: dict) -> str:
    """
    Returns a single <div class='card' data-section='recap'> ... </div> block
    that you can insert into your main page (no new page).
    """
    week, matchup_html, top_by_pos = _matchup_cards_last_week(
        league_id, df_weekly, roster_map, players_map
    )

    sidebar_html = _render_top_three_sidebar(top_by_pos)

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


def build_tables(league_id: str, max_week: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and process league data into DataFrames."""
    # Fetch base data
    users = get_users(league_id)
    rosters = get_rosters(league_id)
    players = get_players_map()

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

    matchups_by_week = build_matchups_by_week(
        league_id, list(range(1, 15)), roster_map, players
    )

    owner_avatar: dict[str, Union[str, None]] = {}
    for r in rosters:
        rid = str(r["roster_id"])
        owner_id = r.get("owner_id")
        display = roster_map.get(rid, f"Roster {rid}")

        avatar_id = None
        if owner_id in user_by_id:
            user_data = user_by_id[owner_id]
            user_meta = user_data.get("metadata") or {}
            u_id = next((u.get("avatar") for u in users if u["user_id"] == r.get("owner_id")), None)
            avatar_id = (
                    user_meta.get("avatar")  # team logo if set
                    or f"https://sleepercdn.com/avatars/{u_id}"  # profile pic fallback
                    or None
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
            weekly_rows.append({
                "week": week,
                "matchup_id": m.get("matchup_id"),
                "roster_id": rid,
                "owner": roster_map.get(rid, f"Roster {rid}"),
                "points": float(m.get("points", 0.0)),
            })
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

    _state = get_nfl_state()
    current_leg = int(_state.get("leg") or _state.get("week") or 0)

    # A week is finalized if the current leg/week is strictly greater
    df_weekly["finalized"] = df_weekly["week"] < current_leg

    # ---- Aggregate stats ----
    records = _compute_team_records(df_weekly[df_weekly["finalized"] == True].copy())
    team_stats = _aggregate_team_stats(df_weekly[df_weekly["finalized"] == True].copy(), records)
    team_stats = team_stats.merge(
        pd.Series(owner_avatar, name="avatar"),
        left_on="owner", right_index=True, how="left",
    )
    last3 = (
        df_weekly[df_weekly["finalized"] == True].copy().sort_values(["owner", "week"])
        .groupby("owner")["points"]
        .apply(lambda s: s.tail(3).mean() if len(s) else 0.0)
        .rename("Last3")
        .reset_index()
    )
    team_stats = team_stats.merge(last3, on="owner", how="left")
    team_stats["Last3"] = team_stats["Last3"].fillna(0.0)
    sos = build_team_strength(team_stats)
    sos = compute_sos_by_team(matchups_by_week, sos, df_weekly.get("week").max(), users)
    sos_df = pd.DataFrame.from_dict(sos, orient='index').reset_index().rename(columns={'index': 'owner'})
    team_stats = team_stats.merge(sos_df, on='owner', how='left')
    streaks_df = compute_streaks(df_weekly[df_weekly["finalized"] == True].copy())
    team_stats = team_stats.merge(streaks_df[df_weekly["finalized"] == True].copy(), on="owner", how="left")
    # Fill empties
    team_stats["StreakType"] = team_stats["StreakType"].fillna("")
    team_stats["StreakLen"] = team_stats["StreakLen"].fillna(0).astype(int)
    team_stats["Streak"] = team_stats["Streak"].fillna("")

    return df_weekly, team_stats, roster_map


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


def build_week_activity(
        league_id: str,
        week: int,
        players_map: Optional[Dict[str, Dict[str, str]]] = None) -> pd.DataFrame:
    """
    Returns rows with structured payloads for HTML:
      kind: 'trade' | 'waiver'
      ts  : datetime (UTC)
      data:
        - trade: {'teams': [{'rid','name','avatar','gets':[players],'sends':[players]}]}
        - waiver: {'rid','name','avatar','adds':[players]}
      (player) -> {'name','pos','team'}
    """
    if players_map is None:
        players_map = get_players_map()

    roster_name, roster_avatar = build_roster_display_maps(league_id)
    txs = fetch_json(f"/league/{league_id}/transactions/{week}") or []
    print(txs)
    rows = []

    def pinfo(pid: str) -> dict[str, str]:
        p = players_map.get(str(pid)) or {}
        return {"name": p.get("name", str(pid)), "pos": p.get("pos", ""), "team": p.get("team", "FA")}

    for t in txs:
        ttype = t.get("type")
        ts = datetime.fromtimestamp((t.get("status_updated") or t.get("created") or 0) / 1000.0, tz=timezone.utc)

        # --- WAIVER ADDS ---
        if ttype in ("waiver", "waiver_add") and isinstance(t.get("adds"), dict):
            # adds: {player_id: roster_id}
            adds = t["adds"]
            by_rid: dict[str, list[dict]] = defaultdict(list)
            for pid, rid in adds.items():
                by_rid[str(rid)].append(pinfo(pid))
            for rid, players in by_rid.items():
                rows.append({
                    "kind": "waiver",
                    "ts": ts,
                    "data": {
                        "rid": rid,
                        "name": roster_name.get(rid, f"Roster {rid}"),
                        "avatar": roster_avatar.get(rid),
                        "adds": players,
                    }
                })
            continue

        # --- TRADES ---
        if ttype == "trade":
            adds = t.get("adds") or {}  # {player_id: roster_id_received}
            drops = t.get("drops") or {}  # {player_id: roster_id_sent}
            team_ids = sorted(set(list(map(str, (t.get("roster_ids") or []))) +
                                  list({str(v) for v in adds.values()}) +
                                  list({str(v) for v in drops.values()})))
            team_objs = []
            for rid in team_ids:
                gets = [pinfo(pid) for pid, to_rid in adds.items() if str(to_rid) == rid]
                sends = [pinfo(pid) for pid, from_rid in drops.items() if str(from_rid) == rid]
                team_objs.append({
                    "rid": rid,
                    "name": roster_name.get(rid, f"Roster {rid}"),
                    "avatar": roster_avatar.get(rid),
                    "gets": gets,
                    "sends": sends
                })
            if team_objs:
                rows.append({"kind": "trade", "ts": ts, "data": {"teams": team_objs}})
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("ts", ascending=False).reset_index(drop=True)
    return df


def build_team_strength(team_stats: Dict[int, dict]) -> Dict[int, float]:
    # e.g., weighted PF: 70% season PF/gm + 30% last-3 PF/gm
    s = {}
    for rid, t in team_stats.iterrows():
        pf_gm = t["PF"] / max(1, t["G"])
        last3 = t["Last3"]
        s[t["owner"]] = round(0.7 * pf_gm + 0.3 * last3, 2)
    return s


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


def compute_sos_by_team(
        all_matchups: Dict[int, List[dict]],  # week -> list of rows
        team_strength: Dict[int, float],
        weeks_past: int,
        users: Dict[int, str],
) -> Dict[int, dict]:
    out = {owner: {"past_sos": 0.0, "past_cnt": 0, "ros_sos": 0.0, "ros_cnt": 0} for owner in team_strength}
    # Past
    for w in range(1, weeks_past):
        for a, b in compute_week_opponents(all_matchups.get(w, [])):
            username = next(
                (u.get("metadata", {}).get("team_name") or a for u in users if u.get("display_name") == a),
                b
            )
            username2 = next(
                (u.get("metadata", {}).get("team_name") or b for u in users if u.get("display_name") == b),
                b
            )
            out[username]["past_sos"] += team_strength[username2];
            out[username]["past_cnt"] += 1
            out[username2]["past_sos"] += team_strength[username];
            out[username2]["past_cnt"] += 1
    # Future (ROS)
    for w in range(weeks_past, 15):
        for a, b in compute_week_opponents(all_matchups.get(w, [])):
            username = next(
                (u.get("metadata", {}).get("team_name") or a for u in users if u.get("display_name") == a),
                b
            )
            username2 = next(
                (u.get("metadata", {}).get("team_name") or b for u in users if u.get("display_name") == b),
                b
            )
            out[username]["ros_sos"] += team_strength[username2];
            out[username]["ros_cnt"] += 1
            out[username2]["ros_sos"] += team_strength[username];
            out[username2]["ros_cnt"] += 1
    # Averages
    for rid, v in out.items():
        v["past_sos"] = v["past_sos"] / v["past_cnt"] if v["past_cnt"] else 0.0
        v["ros_sos"] = v["ros_sos"] / v["ros_cnt"] if v["ros_cnt"] else 0.0
    return out
