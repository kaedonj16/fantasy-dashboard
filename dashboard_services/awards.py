import pandas as pd

from .api import get_matchups
from .players import build_roster_map


def render_awards_section(awards: dict) -> str:
    if not awards:
        return "<div class='card' data-section='awards'><h2>Awards</h2><div class='muted'>No data</div></div>"

    def acard(title, body):
        return f"""
        <div class="award-item">
          <div class="award-name">{title}</div>
          <div class="award-body">{body}</div>
        </div>"""

    rows = []

    if awards.get("highest_single_week"):
        t, w, p = awards["highest_single_week"]
        rows.append(acard("Highest Single Week", f"{t} — Week {w}: <strong>{p:.1f} points</strong>"))

    if awards.get("lowest_single_week"):
        t, w, p = awards["lowest_single_week"]
        rows.append(acard("Lowest Single Week", f"{t} — Week {w}: <strong>{p:.1f} points</strong>"))

    if awards.get("longest_win_streak"):
        t, L = awards["longest_win_streak"]
        rows.append(acard("Longest Win Streak", f"{t} — <strong>{L} games</strong>"))

    if awards.get("longest_loss_streak"):
        t, L = awards["longest_loss_streak"]
        rows.append(acard("Longest Losing Streak", f"{t} — <strong>{L} games</strong>"))

    if awards.get("most_consistent"):
        t, sd, n = awards["most_consistent"]
        rows.append(acard("Most Consistent", f"{t} — σ <strong>{sd:.2f}</strong> over {n} games"))

    if awards.get("highest_player"):
        w, pts, n, pos, team, owner = awards["highest_player"]
        rows.append(acard("Highest Points By a Player", f"{n} — Week {w}: <strong>{pts} points</strong>"))

    return f"""
    <div class="card awards-card" data-section="awards">
        <h2 class="awards-title">🏆 League Awards</h2>
      <div class="awards-grid">{''.join(rows)}</div>
    </div>
    """


def highest_single_game_points(league_id: str,
                               players_map: dict,
                               weeks=range(1, 18),) -> dict:
    """
    Returns a dict for the single highest fantasy score by any NFL player in your league:
      {
        'week', 'points', 'name', 'pos', 'nfl', 'owner'
      }

    Set started_only=True to consider only players that were in starting lineups.
    """
    # You likely already have this helper; otherwise map roster_id -> "Team (Owner)"
    roster_map = build_roster_map(league_id)
    best = ["", 0.0, "", "", "", ""]


    for w in range(1, weeks):
        matchups = get_matchups(league_id, w) or []
        for row in matchups:
            rid = row.get("roster_id")
            owner = roster_map.get(str(rid), f"Roster {rid}")
            ppts = row.get("players_points") or {}
            for pid_str, pts in ppts.items():
                pts_f = float(pts or 0)

                if pts_f > best[1]:
                    p = players_map.get(pid_str, {})
                    best = [
                        w,
                        pts_f,
                        p.get("name"),
                        p.get("position"),
                        p.get("team") or "FA",
                        owner,
                    ]

    # If nothing found (e.g., empty season), normalize to None/0
    if best[0] is None:
        return {
            None, 0.0,
            None, None, None,
            None
        }
    return best


def compute_awards_season(df_weekly: pd.DataFrame, players_map: dict, league_id: str) -> dict:
    """
    Returns a dict with keys mapping to tuples of display-friendly values.
      highest_single_week: (team, week, points)
      lowest_single_week:  (team, week, points)
      longest_win_streak:  (team, length)
      longest_loss_streak: (team, length)
      most_consistent:     (team, std_dev, games_played)
      highest_ceiling:     (team, max_points)
    Requires columns: Week, Team, Opponent, Points, OppPoints
    """
    d = {}

    # Single-week extremes
    idx_hi = df_weekly["points"].idxmax()
    idx_lo = df_weekly["points"].idxmin()
    d["highest_single_week"] = (
        df_weekly.loc[idx_hi, "owner"],
        int(df_weekly.loc[idx_hi, "week"]),
        float(df_weekly.loc[idx_hi, "points"])
    )
    d["lowest_single_week"] = (
        df_weekly.loc[idx_lo, "owner"],
        int(df_weekly.loc[idx_lo, "week"]),
        float(df_weekly.loc[idx_lo, "points"])
    )

    # Per-team sequences for streaks
    def longest_streak(arr, kind="+"):
        best = cur = 0
        for x in arr:
            ok = (x == kind)
            cur = cur + 1 if ok else 0
            best = max(best, cur)
        return best

    # Build W/L array per team based on points
    wl_map = {}
    for t, g in df_weekly.groupby("owner"):
        g = g.sort_values("week")
        seq = []
        for _, r in g.iterrows():
            if r["points"] > r["points_against"]:
                seq.append("+")  # win
            elif r["points"] < r["points_against"]:
                seq.append("-")  # loss
            else:
                seq.append("=")  # tie
        wl_map[t] = seq

    # Longest streaks
    win_streaks = [(t, longest_streak(seq, "+")) for t, seq in wl_map.items()]
    loss_streaks = [(t, longest_streak(seq, "-")) for t, seq in wl_map.items()]
    d["longest_win_streak"] = max(win_streaks, key=lambda x: x[1]) if win_streaks else None
    d["longest_loss_streak"] = max(loss_streaks, key=lambda x: x[1]) if loss_streaks else None

    # Consistency (std dev of points) — require at least 4 games to be fair
    cons = []
    for t, g in df_weekly.groupby("owner"):
        pts = g["points"].astype(float)
        if len(pts) >= 4:
            cons.append((t, float(pts.std(ddof=0)), int(len(pts))))
    d["most_consistent"] = min(cons, key=lambda x: x[1]) if cons else None

    best_started = highest_single_game_points(league_id, players_map, 18)
    d["highest_player"] = best_started

    return d


def compute_weekly_highlights(df_weekly: pd.DataFrame, week: int) -> dict:
    """
    Returns a dict with:
      - top_team: (team, points)
      - closest: (teamA, teamB, diff, scoreA, scoreB)
      - blowout: (teamA, teamB, diff, scoreA, scoreB)  # diff is abs margin
    """
    w = df_weekly[df_weekly["week"] == week].copy()
    if w.empty:
        return {}

    # Highest-scoring team
    idx_max = w["points"].idxmax()
    idx_min = w["points"].idxmin()
    top_team = (w.loc[idx_max, "owner"], float(w.loc[idx_max, "points"]))
    min_team = (w.loc[idx_min, "owner"], float(w.loc[idx_min, "points"]))

    # Build matchups by pairing Team/Opponent rows
    # Key each game by frozenset({Team, Opponent})
    w["_pair"] = w.apply(lambda r: frozenset({r["owner"], r["matchup_id"]}), axis=1)
    # For each pair, take two rows (team & opponent), compute margin
    games = []
    for pair, g in w.groupby("matchup_id"):
        if len(g) != 2:
            continue
        a = g.iloc[0]
        b = g.iloc[1]
        # Normalize so A is the one listed in 'Team' from first row
        teamA, ptsA, teamB, ptsB = a["owner"], float(a["points"]), b["owner"], float(b["points"])
        diff = abs(ptsA - ptsB)
        games.append((teamA, teamB, diff, ptsA, ptsB))

    if not games:
        closest = blowout = None
    else:
        games_sorted = sorted(games, key=lambda x: x[2])  # by diff
        closest = games_sorted[0]
        blowout = sorted(games, key=lambda x: x[2], reverse=True)[0]

    return {"top_team": top_team, "low_team": min_team, "closest": closest, "blowout": blowout}
