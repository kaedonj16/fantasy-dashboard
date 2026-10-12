import html
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .platform_api import get_matchups, get_rosters, get_users
from .players import build_roster_map

logger = logging.getLogger(__name__)

# All-time awards cache: historical seasons are immutable, so cache
# aggressively. Keyed (platform, league_id).
_ALLTIME_AWARDS_CACHE: dict = {}
_ALLTIME_AWARDS_TTL = 3600
_ALLTIME_MAX_SEASONS = 5


def render_awards_section(awards: dict, alltime: bool = False) -> str:
    if not awards:
        # Offseason / week 0: hide empty weekly superlatives. All-time records
        # live on the Awards page.
        return ""

    def _week_detail(w, s=None):
        base = f"Week {w}"
        if s is not None:
            base += f" \u00b7 {s}"
        return base

    def arow(label, team_html, detail, value_html):
        _detail = (f"<div class=\"cc-award-detail\">{detail}</div>"
                   if detail else "")
        return (
            f"<div class=\"cc-award\">"
            f"<div class=\"cc-award-info\">"
            f"<div class=\"cc-award-label\">{label}</div>"
            f"<div class=\"cc-award-team\">{team_html}</div>"
            f"{_detail}</div>"
            f"{value_html}</div>"
        )

    def aval(num, color=""):
        cls = f"cc-award-val{(' ' + color) if color else ''}"
        return f"<div class=\"{cls}\">{num}</div>"

    def streak_winner(teams):
        # Tied teams stack one per line; a comma run-on reads as one name.
        _esc = [html.escape(str(t)) for t in (teams or [])]
        if len(_esc) > 1:
            return "<br>".join(_esc)
        return _esc[0] if _esc else ""

    rows = []

    if awards.get("highest_single_week"):
        tup = awards["highest_single_week"]
        t, w, p = tup[0], tup[1], tup[2]
        s = tup[3] if len(tup) > 3 else None
        rows.append(arow("Highest single week", html.escape(str(t)),
                         _week_detail(w, s), aval(f"{p:.1f}", "cc-up")))

    if awards.get("lowest_single_week"):
        tup = awards["lowest_single_week"]
        t, w, p = tup[0], tup[1], tup[2]
        s = tup[3] if len(tup) > 3 else None
        rows.append(arow("Lowest single week", html.escape(str(t)),
                         _week_detail(w, s), aval(f"{p:.1f}", "cc-dn")))

    if awards.get("longest_win_streak"):
        teams, L = awards["longest_win_streak"]
        rows.append(arow("Longest win streak", streak_winner(teams),
                         "", aval(f"{L}", "cc-up")))

    if awards.get("longest_loss_streak"):
        teams, L = awards["longest_loss_streak"]
        rows.append(arow("Longest losing streak", streak_winner(teams),
                         "", aval(f"{L}", "cc-dn")))

    if awards.get("most_consistent"):
        t, sd, n = awards["most_consistent"]
        rows.append(arow("Most consistent", html.escape(str(t)),
                         f"over {n} games", aval(f"{sd:.2f}")))

    if awards.get("highest_player"):
        hp = awards["highest_player"]
        w, pts, n, pos, team, owner, pid = hp[0], hp[1], hp[2], hp[3], hp[4], hp[5], hp[6]
        s = hp[7] if len(hp) > 7 else None
        _pn = html.escape(str(n))
        clickable_attrs = f" class='player-clickable' style='cursor:pointer;' data-player-id='{html.escape(str(pid), quote=True)}' data-player-name='{_pn}'" if pid else ""
        rows.append(arow("Top player week",
                         f"<span{clickable_attrs}>{_pn}</span>",
                         _week_detail(w, s), aval(f"{pts:.1f}", "cc-up")))

    sub = "All-time league history" if alltime else "Season superlatives so far"
    return f"""
    <section class="os-side-plain cc-aw-card" data-section="awards">
      <div class="cc-aw-head">
        <div class="cc-aw-head-icon"><i class="fa-solid fa-trophy" aria-hidden="true"></i></div>
        <div><h3 class="cc-aw-title">League Awards</h3></div>
        <div class="cc-aw-sub">{sub}</div>
        <button type="button" class="card-collapse-toggle cc-head-toggle" aria-label="Toggle section" aria-expanded="true" data-target="dash-awards-body">&#9660;</button>
      </div>
      <div class="card-collapsible-body" id="dash-awards-body">
        <div class="cc-aw-body">{''.join(rows)}</div>
      </div>
    </section>
    """


def highest_single_game_points(league_id: str,
                               players_map: dict,
                               platform: str,
                               season,
                               users,
                               rosters,
                               weeks=range(1, 18), ) -> dict:
    """
    Returns a dict for the single highest fantasy score by any NFL player in your league:
      {
        'week', 'points', 'name', 'pos', 'nfl', 'owner', 'pid'
      }

    Set started_only=True to consider only players that were in starting lineups.
    """
    # You likely already have this helper; otherwise map roster_id -> "Team (Owner)"
    roster_map = build_roster_map(platform, league_id, season, users, rosters)
    best = ["", 0.0, "", "", "", "", ""]

    for w in range(1, weeks):
        try:
            matchups = get_matchups(platform, league_id, w, season) or []
        except Exception:
            # A single failed week (e.g. a Fleaflicker edge/WAF HTML 403 on
            # FetchLeagueScoreboard) must not 503 the whole dashboard.
            # Score the weeks that did load and move on.
            logger.debug(
                "highest_single_game_points: week %s unavailable platform=%s league=%s; skipping",
                w, platform, league_id, exc_info=True,
            )
            continue
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
                        pid_str,
                    ]

    # If nothing found (e.g., empty season), normalize to None/0
    if best[0] is None:
        return {
            None, 0.0,
            None, None, None,
            None, None
        }
    return best


def _awards_core(df_weekly: pd.DataFrame) -> dict:
    """Shared df-driven awards (no player-week scan).

    Requires columns: owner, week, points, points_against. When a ``season``
    column is present, week extremes carry it as a 4th tuple element so the
    renderer can label the season.
    """
    d = {}

    # No finalized games yet (offseason / week 0) means there's nothing to award.
    # idxmax() on the empty (or all-NaN) points column raises, so bail early to a
    # "No data" awards card instead of 500'ing the whole dashboard.
    if (df_weekly is None or df_weekly.empty
            or "points" not in df_weekly.columns
            or not df_weekly["points"].notna().any()):
        return {}

    has_season = "season" in df_weekly.columns

    def _season_of(row):
        return (int(row["season"]),) if has_season else ()

    # Single-week extremes
    idx_hi = df_weekly["points"].idxmax()
    idx_lo = df_weekly["points"].idxmin()
    r_hi = df_weekly.loc[idx_hi]
    r_lo = df_weekly.loc[idx_lo]
    d["highest_single_week"] = (
        r_hi["owner"],
        int(r_hi["week"]),
        float(r_hi["points"]),
    ) + _season_of(r_hi)
    d["lowest_single_week"] = (
        r_lo["owner"],
        int(r_lo["week"]),
        float(r_lo["points"]),
    ) + _season_of(r_lo)

    # Per-team sequences for streaks
    def longest_streak(arr, kind="+"):
        best = cur = 0
        for x in arr:
            ok = (x == kind)
            cur = cur + 1 if ok else 0
            best = max(best, cur)
        return best

    # Build W/L array per team based on points
    sort_keys = ["season", "week"] if has_season else ["week"]
    wl_map = {}
    for t, g in df_weekly.groupby("owner"):
        g = g.sort_values(sort_keys)
        seq = []
        for _, r in g.iterrows():
            if r["points"] > r["points_against"]:
                seq.append("+")  # win
            elif r["points"] < r["points_against"]:
                seq.append("-")  # loss
            else:
                seq.append("=")  # tie
        wl_map[t] = seq

    # Longest streaks — keep every team tied at the top, drop zero-length streaks,
    # and skip blank team names so the card never renders empty.
    def _top_streaks(streaks):
        named = [(t, L) for t, L in streaks if t and str(t).strip()]
        top = max((L for _, L in named), default=0)
        if top <= 0:
            return None
        tied = sorted(str(t) for t, L in named if L == top)
        return (tied, top)

    win_streaks = [(t, longest_streak(seq, "+")) for t, seq in wl_map.items()]
    loss_streaks = [(t, longest_streak(seq, "-")) for t, seq in wl_map.items()]
    d["longest_win_streak"] = _top_streaks(win_streaks)
    d["longest_loss_streak"] = _top_streaks(loss_streaks)

    # Consistency (std dev of points) - require at least 4 games to be fair
    cons = []
    for t, g in df_weekly.groupby("owner"):
        pts = g["points"].astype(float)
        if len(pts) >= 4:
            cons.append((t, float(pts.std(ddof=0)), int(len(pts))))
    d["most_consistent"] = min(cons, key=lambda x: x[1]) if cons else None

    return d


def compute_awards_season(df_weekly: pd.DataFrame, players_map: dict, league_id: str, platform: str, season: str, users,
                          rosters) -> dict:
    """
    Single-season awards. Returns a dict with keys mapping to tuples of
    display-friendly values.
      highest_single_week: (team, week, points)
      lowest_single_week:  (team, week, points)
      longest_win_streak:  ([teams], length) — every team tied at the max, None if no positive streak
      longest_loss_streak: ([teams], length) — every team tied at the max, None if no positive streak
      most_consistent:     (team, std_dev, games_played)
      highest_player:      (week, points, name, pos, team, owner, pid)
    Requires columns: owner, week, points, points_against
    """
    d = _awards_core(df_weekly)
    if not d:
        return d
    best_started = highest_single_game_points(league_id, players_map, platform, season, users, rosters, 18)
    d["highest_player"] = best_started
    return d


def _add_points_against(df: pd.DataFrame) -> pd.DataFrame:
    """Fill points_against by pairing the two rows of each (week, matchup_id)."""
    df = df.copy()
    df["points_against"] = float("nan")
    if df.empty or "matchup_id" not in df.columns:
        return df
    for (_, _mid), grp in df.groupby(["week", "matchup_id"]):
        if len(grp) == 2:
            i1, i2 = grp.index.tolist()
            df.loc[i1, "points_against"] = df.loc[i2, "points"]
            df.loc[i2, "points_against"] = df.loc[i1, "points"]
    return df


def _fetch_season_week_rows(platform: str, league_id: str, season: int, roster_map: dict) -> list:
    """One season of weekly team rows. Parallel week fetch, like service.py."""
    rows: list = []

    def _fetch_week(w: int, _rm: dict = roster_map) -> list:
        out: list = []
        try:
            wk = get_matchups(platform, league_id, w, season) or []
        except Exception:
            return out
        for m in wk:
            rid = str(m.get("roster_id"))
            out.append({
                "week": w,
                "season": season,
                "owner": _rm.get(rid, f"Roster {rid}"),
                "points": float(m.get("points") or 0.0),
                "matchup_id": m.get("matchup_id"),
                "players_points": m.get("players_points") or {},
            })
        return out

    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(_fetch_week, w) for w in range(1, 18)]
        for fut in as_completed(futs):
            try:
                rows.extend(fut.result())
            except Exception:
                logger.debug("all-time awards: week fetch failed", exc_info=True)
    return rows


def _compute_awards_alltime(platform: str, league_id: str, season: int,
                            players_map: dict, users, rosters, finalized_df) -> dict:
    """Uncached all-time awards. Raises on unexpected failure; caller catches."""
    from .api import build_league_history_map  # deferred: api is heavyweight

    history = build_league_history_map(platform, league_id, season) or {}
    seasons = sorted(history.keys(), reverse=True)[:_ALLTIME_MAX_SEASONS]
    if not seasons:
        return {}

    frames: list = []
    best_player = None  # [week, points, pid, owner_label, season]

    # Current season: reuse the dashboard's finalized frame (no refetch).
    cur = finalized_df
    if cur is not None and not getattr(cur, "empty", True):
        cols = [c for c in ("owner", "week", "points", "points_against") if c in cur.columns]
        if len(cols) == 4:
            f = cur[cols].copy()
            f["season"] = season
            frames.append(f)

    for s in seasons:
        if s == season:
            continue
        lid = str(history[s])
        try:
            users_s = get_users(platform, lid, s) or []
            rosters_s = get_rosters(platform, lid, s) or []
        except Exception:
            logger.debug("all-time awards: season %s users/rosters failed", s, exc_info=True)
            continue
        if not rosters_s:
            continue
        roster_map_s = build_roster_map(lid, platform, s, users_s, rosters_s)

        rows = _fetch_season_week_rows(platform, lid, s, roster_map_s)
        if not rows:
            continue

        # Top player-week across all seasons, tracked during the same fetch.
        for r in rows:
            _owner = r["owner"]
            for pid_str, pts in (r["players_points"] or {}).items():
                pts_f = float(pts or 0)
                if best_player is None or pts_f > best_player[1]:
                    best_player = [r["week"], pts_f, str(pid_str), _owner, r["season"]]

        df_s = _add_points_against(pd.DataFrame(rows))
        frames.append(df_s[["owner", "week", "points", "points_against", "season"]])

    if not frames:
        return {}
    all_df = pd.concat(frames, ignore_index=True)
    d = _awards_core(all_df)
    if not d:
        return {}

    # Top player week: best of the current-season scan and the prior-season scan.
    candidates: list = []
    try:
        cur_best = highest_single_game_points(league_id, players_map, platform, season, users, rosters, 18)
    except Exception:
        logger.debug("all-time awards: current-season player scan failed", exc_info=True)
        cur_best = None
    if cur_best and cur_best[0]:
        candidates.append(
            (cur_best[0], cur_best[1], cur_best[2], cur_best[3],
             cur_best[4], cur_best[5], cur_best[6], season)
        )
    if best_player is not None:
        w, pts, pid, owner, s = best_player
        p = (players_map or {}).get(str(pid), {}) or {}
        candidates.append(
            (w, pts, p.get("name"), p.get("position"), p.get("team") or "FA",
             owner, str(pid), s)
        )
    d["highest_player"] = max(candidates, key=lambda c: float(c[1] or 0)) if candidates else None
    return d


def compute_awards_alltime(platform: str, league_id: str, season, players_map: dict,
                           users, rosters, finalized_df) -> dict:
    """All-time league awards across seasons (newest-first, capped).

    Historical seasons are immutable, so results are cached for an hour.
    Never raises: returns {} on any failure so the caller can fall back to
    single-season awards.
    """
    try:
        key = (str(platform or "").lower(), str(league_id or ""))
        now = time.time()
        hit = _ALLTIME_AWARDS_CACHE.get(key)
        if hit and now - hit[0] < _ALLTIME_AWARDS_TTL:
            return hit[1]
        result = _compute_awards_alltime(platform, league_id, int(season),
                                         players_map, users, rosters, finalized_df)
        _ALLTIME_AWARDS_CACHE[key] = (now, result)
        return result
    except Exception:
        logger.debug("compute_awards_alltime failed", exc_info=True)
        return {}

