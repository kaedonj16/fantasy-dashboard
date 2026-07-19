from __future__ import annotations

from typing import Any, Dict, List

import math
from html import escape as _esc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from flask import url_for
from plotly.offline import plot as plotly_plot

from dashboard_services.ai.history_recap import get_league_season_summary
from dashboard_services.platform_api import get_bracket
from utils.coerce import safe_int as _safe_int


def _playoff_start_week(league: dict) -> int:
    settings = league.get("settings") or {}
    try:
        return int(settings.get("playoff_week_start") or 15)
    except (TypeError, ValueError):
        return 15


def build_regular_season_team_stats(
        df_weekly: pd.DataFrame,
        league: dict,
) -> pd.DataFrame:
    """
    Recompute standings/stats using only regular season weeks.
    This avoids playoff results changing the year recap standings.
    """
    if df_weekly is None or df_weekly.empty:
        return pd.DataFrame()

    playoff_start = _playoff_start_week(league)

    df = df_weekly.copy()

    if "week" not in df.columns or "owner" not in df.columns or "points" not in df.columns:
        return pd.DataFrame()

    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0.0)

    # regular season only
    df = df[df["week"] < playoff_start]

    # finalized only if available
    if "finalized" in df.columns:
        finalized_df = df[df["finalized"] == True].copy()
        if not finalized_df.empty:
            df = finalized_df

    if df.empty:
        return pd.DataFrame()

    rows = []

    for owner, team_df in df.groupby("owner"):
        team_df = team_df.sort_values(["week", "matchup_id"] if "matchup_id" in team_df.columns else ["week"])

        wins = 0
        losses = 0
        ties = 0
        pf = float(team_df["points"].sum())
        pa = 0.0

        if "matchup_id" in team_df.columns:
            for (week, matchup_id), grp in df[df["week"].isin(team_df["week"].unique())].groupby(
                    ["week", "matchup_id"]):
                if len(grp) != 2:
                    continue

                grp = grp.copy()
                grp["points"] = pd.to_numeric(grp["points"], errors="coerce").fillna(0.0)

                owner_rows = grp[grp["owner"] == owner]
                if owner_rows.empty:
                    continue

                my_row = owner_rows.iloc[0]
                opp_row = grp[grp["owner"] != owner]
                if opp_row.empty:
                    continue
                opp_row = opp_row.iloc[0]

                my_pts = float(my_row["points"])
                opp_pts = float(opp_row["points"])

                pa += opp_pts

                if my_pts > opp_pts:
                    wins += 1
                elif my_pts < opp_pts:
                    losses += 1
                else:
                    ties += 1
        else:
            # fallback if matchup_id is unavailable
            pa = 0.0

        games = wins + losses + ties
        avg = pf / games if games > 0 else 0.0
        std = float(team_df["points"].std()) if len(team_df) > 1 else 0.0
        best = float(team_df["points"].max()) if not team_df.empty else 0.0
        worst = float(team_df["points"].min()) if not team_df.empty else 0.0
        win_pct = wins / games if games > 0 else 0.0

        rows.append(
            {
                "owner": owner,
                "Wins": wins,
                "Losses": losses,
                "Ties": ties,
                "Win%": win_pct,
                "PF": pf,
                "PA": pa,
                "AVG": avg,
                "STD": std,
                "MAX": best,
                "MIN": worst,
                "Streak": "",
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(
        ["Wins", "Win%", "PF", "PA"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    out["Rank"] = out.index + 1
    return out


def _build_full_season_stats(
        df_weekly: pd.DataFrame,
        league: dict,
) -> pd.DataFrame:
    """
    Build stats for the entire season including playoffs.
    Similar to build_regular_season_team_stats but includes all weeks.
    """
    if df_weekly is None or df_weekly.empty:
        return pd.DataFrame()

    df = df_weekly.copy()

    if "week" not in df.columns or "owner" not in df.columns or "points" not in df.columns:
        return pd.DataFrame()

    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0.0)

    # Include ALL weeks (regular season + playoffs)
    # No filtering by playoff_start_week

    # finalized only if available
    if "finalized" in df.columns:
        finalized_df = df[df["finalized"] == True].copy()
        if not finalized_df.empty:
            df = finalized_df

    if df.empty:
        return pd.DataFrame()

    rows = []

    for owner, team_df in df.groupby("owner"):
        team_df = team_df.sort_values(["week", "matchup_id"] if "matchup_id" in team_df.columns else ["week"])

        wins = 0
        losses = 0
        ties = 0
        pf = float(team_df["points"].sum())
        pa = 0.0

        if "matchup_id" in team_df.columns:
            for (week, matchup_id), grp in df[df["week"].isin(team_df["week"].unique())].groupby(
                    ["week", "matchup_id"]):
                if len(grp) != 2:
                    continue

                grp = grp.copy()
                grp["points"] = pd.to_numeric(grp["points"], errors="coerce").fillna(0.0)

                owner_rows = grp[grp["owner"] == owner]
                if owner_rows.empty:
                    continue

                my_row = owner_rows.iloc[0]
                opp_row = grp[grp["owner"] != owner]
                if opp_row.empty:
                    continue
                opp_row = opp_row.iloc[0]

                my_pts = float(my_row["points"])
                opp_pts = float(opp_row["points"])

                pa += opp_pts

                if my_pts > opp_pts:
                    wins += 1
                elif my_pts < opp_pts:
                    losses += 1
                else:
                    ties += 1
        else:
            # fallback if matchup_id is unavailable
            pa = 0.0

        games = wins + losses + ties
        avg = pf / games if games > 0 else 0.0
        std = float(team_df["points"].std()) if len(team_df) > 1 else 0.0
        best = float(team_df["points"].max()) if not team_df.empty else 0.0
        worst = float(team_df["points"].min()) if not team_df.empty else 0.0
        win_pct = wins / games if games > 0 else 0.0

        rows.append(
            {
                "owner": owner,
                "Wins": wins,
                "Losses": losses,
                "Ties": ties,
                "Win%": win_pct,
                "PF": pf,
                "PA": pa,
                "AVG": avg,
                "STD": std,
                "MAX": best,
                "MIN": worst,
                "Streak": "",
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(
        ["Wins", "Win%", "PF", "PA"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    out["Rank"] = out.index + 1
    return out


def _team_stats_lookup(team_stats: pd.DataFrame, team_name: str) -> dict:
    if team_stats is None or team_stats.empty or not team_name or team_name == "-":
        return {}

    df = team_stats.copy()
    if "owner" not in df.columns:
        return {}

    match = df[df["owner"].astype(str) == str(team_name)]
    if match.empty:
        return {}

    row = match.iloc[0]
    return {
        "Wins": _safe_int(row.get("Wins"), 0),
        "Losses": _safe_int(row.get("Losses"), 0),
        "Ties": _safe_int(row.get("Ties"), 0),
        "PF": _safe_float(row.get("PF"), 0.0),
        "PA": _safe_float(row.get("PA"), 0.0),
        "AVG": _safe_float(row.get("AVG"), 0.0),
    }


def get_champion_and_runner_up(ctx: dict) -> tuple[str, str]:
    """
    Pull champion and runner-up from the winners bracket.
    Prefers the championship game (p == 1) when Sleeper provides it.
    """
    platform = ctx["platform"]
    season = int(ctx["season"])
    league_id = ctx.get("resolved_league_id") or ctx["league_id"]
    roster_map = ctx.get("roster_map") or {}

    try:
        # Adjust this call if your wrapper signature differs in your current repo.
        winners_bracket = get_bracket(platform, league_id, "winners", season) or []
    except Exception:
        winners_bracket = []

    if not winners_bracket:
        return "-", "-"

    # Only consider completed matchups
    completed = [
        m for m in winners_bracket
        if m.get("w") is not None and m.get("l") is not None
    ]
    if not completed:
        return "-", "-"

    # Best case: Sleeper marks the championship game with p == 1
    championship = next(
        (m for m in completed if _safe_int(m.get("p"), 0) == 1),
        None,
    )

    if championship is None:
        # Fallback: take the deepest round, then lowest placement number if present,
        # then lowest match id.
        completed = sorted(
            completed,
            key=lambda m: (
                _safe_int(m.get("r"), 0),  # deepest round
                -(_safe_int(m.get("p"), 999) or 999),  # prefer p=1 over p=3, etc.
                -_safe_int(m.get("m"), 0),  # stable fallback
            ),
            reverse=True,
        )
        championship = completed[0]

    winner_id = championship.get("w")
    loser_id = championship.get("l")

    champion = (
            roster_map.get(str(winner_id))
            or roster_map.get(winner_id)
            or f"Roster {winner_id}"
    )
    runner_up = (
            roster_map.get(str(loser_id))
            or roster_map.get(loser_id)
            or f"Roster {loser_id}"
    )

    return str(champion), str(runner_up)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        v = float(value)
        if math.isnan(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _record_str(row: pd.Series) -> str:
    wins = _safe_int(row.get("Wins"), 0)
    losses = _safe_int(row.get("Losses"), 0)
    ties = _safe_int(row.get("Ties"), 0)
    if ties:
        return f"{wins}-{losses}-{ties}"
    return f"{wins}-{losses}"


def sort_team_stats(team_stats: pd.DataFrame) -> pd.DataFrame:
    if team_stats is None or team_stats.empty:
        return pd.DataFrame()

    df = team_stats.copy()

    if "Wins" not in df.columns:
        df["Wins"] = 0
    if "Losses" not in df.columns:
        df["Losses"] = 0
    if "Ties" not in df.columns:
        df["Ties"] = 0
    if "PF" not in df.columns:
        df["PF"] = 0.0
    if "PA" not in df.columns:
        df["PA"] = 0.0
    if "AVG" not in df.columns:
        df["AVG"] = 0.0
    if "STD" not in df.columns:
        df["STD"] = 0.0

    df["Wins"] = pd.to_numeric(df["Wins"], errors="coerce").fillna(0)
    df["Losses"] = pd.to_numeric(df["Losses"], errors="coerce").fillna(0)
    df["Ties"] = pd.to_numeric(df["Ties"], errors="coerce").fillna(0)
    df["PF"] = pd.to_numeric(df["PF"], errors="coerce").fillna(0.0)
    df["PA"] = pd.to_numeric(df["PA"], errors="coerce").fillna(0.0)
    df["AVG"] = pd.to_numeric(df["AVG"], errors="coerce").fillna(0.0)
    df["STD"] = pd.to_numeric(df["STD"], errors="coerce").fillna(0.0)

    games = df["Wins"] + df["Losses"] + df["Ties"]
    df["WinPct"] = np.where(games > 0, df["Wins"] / games, 0.0)

    owner_col = "owner"
    if owner_col not in df.columns:
        if "Team" in df.columns:
            owner_col = "Team"
        else:
            df["owner"] = [f"Team {i + 1}" for i in range(len(df))]
            owner_col = "owner"

    sort_cols = ["Wins", "WinPct", "PF", "PA"]
    ascending = [False, False, False, True]

    df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    df["Rank"] = df.index + 1

    if owner_col != "owner":
        df["owner"] = df[owner_col].astype(str)

    return df


def _team_name_from_roster_id(roster_map: Dict[str, str], roster_id: Any) -> str:
    if roster_id is None:
        return "-"
    return (
            roster_map.get(str(roster_id))
            or roster_map.get(roster_id)
            or f"Roster {roster_id}"
    )


def _filtered_season_df(df_weekly: pd.DataFrame) -> pd.DataFrame:
    if df_weekly is None or df_weekly.empty:
        return pd.DataFrame()

    df = df_weekly.copy()

    if "finalized" in df.columns:
        finalized = df[df["finalized"] == True].copy()
        if not finalized.empty:
            df = finalized

    if "points" in df.columns:
        df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0.0)

    if "week" in df.columns:
        df["week"] = pd.to_numeric(df["week"], errors="coerce").fillna(0).astype(int)

    return df


def _build_summary(history_ctx: dict) -> dict:
    league = history_ctx.get("league") or {}
    df_weekly = _filtered_season_df(history_ctx.get("df_weekly", pd.DataFrame()))

    # regular season standings only
    team_stats = build_regular_season_team_stats(df_weekly, league)
    team_stats = sort_team_stats(team_stats)

    # Build full season stats for champion/runner-up records (including playoffs)
    full_season_stats = _build_full_season_stats(df_weekly, league)

    champion, runner_up = get_champion_and_runner_up(history_ctx)
    # Use full season stats for champion/runner-up records
    champion_stats = _team_stats_lookup(full_season_stats, champion)
    runner_up_stats = _team_stats_lookup(full_season_stats, runner_up)

    summary = {
        "champion": champion,
        "champion_record": _record_str(pd.Series(champion_stats)) if champion_stats else "-",
        "runner_up": runner_up,
        "runner_up_record": _record_str(pd.Series(runner_up_stats)) if runner_up_stats else "-",
        "top_scorer_team": "-",
        "top_scorer_value": 0.0,
        "top_scorer_avg": 0.0,
        "best_defense_team": "-",
        "best_defense_value": 0.0,
        "highest_week_team": "-",
        "highest_week_value": 0.0,
        "lowest_week_team": "-",
        "lowest_week_value": 0.0,
        "closest_matchup": "-",
        "closest_margin": 0.0,
        "biggest_blowout": "-",
        "biggest_blowout_margin": 0.0,
        "unluckiest_team": "-",
        "unluckiest_delta": 0,
    }

    if not team_stats.empty:
        pf_idx = team_stats["PF"].idxmax()
        pa_idx = team_stats["PA"].idxmin()

        summary["top_scorer_team"] = str(team_stats.loc[pf_idx, "owner"])
        summary["top_scorer_value"] = _safe_float(team_stats.loc[pf_idx, "PF"])
        summary["top_scorer_avg"] = _safe_float(team_stats.loc[pf_idx, "AVG"])

        summary["best_defense_team"] = str(team_stats.loc[pa_idx, "owner"])
        summary["best_defense_value"] = _safe_float(team_stats.loc[pa_idx, "PA"])

        pf_rank = (
            team_stats[["owner", "PF"]]
            .sort_values("PF", ascending=False)
            .reset_index(drop=True)
        )
        pf_rank["pf_rank"] = pf_rank.index + 1

        win_rank = (
            team_stats[["owner", "Wins", "PF"]]
            .sort_values(["Wins", "PF"], ascending=[False, False])
            .reset_index(drop=True)
        )
        win_rank["win_rank"] = win_rank.index + 1

        merged = pf_rank.merge(win_rank[["owner", "win_rank"]], on="owner", how="inner")
        merged["delta"] = merged["win_rank"] - merged["pf_rank"]

        if not merged.empty:
            unlucky = merged.sort_values(["delta", "pf_rank"], ascending=[False, True]).iloc[0]
            summary["unluckiest_team"] = str(unlucky["owner"])
            summary["unluckiest_delta"] = _safe_int(unlucky["delta"], 0)

    if not df_weekly.empty and {"owner", "points"}.issubset(df_weekly.columns):
        hi = df_weekly.loc[df_weekly["points"].idxmax()]
        lo = df_weekly.loc[df_weekly["points"].idxmin()]

        summary["highest_week_team"] = str(hi.get("owner", "-"))
        summary["highest_week_value"] = _safe_float(hi.get("points"))

        summary["lowest_week_team"] = str(lo.get("owner", "-"))
        summary["lowest_week_value"] = _safe_float(lo.get("points"))

    if not df_weekly.empty and {"week", "matchup_id", "owner", "points"}.issubset(df_weekly.columns):
        matchup_rows = []
        for (_, _), grp in df_weekly.groupby(["week", "matchup_id"]):
            if len(grp) != 2:
                continue

            ordered = grp.sort_values("points", ascending=False).reset_index(drop=True)
            winner = ordered.iloc[0]
            loser = ordered.iloc[1]
            margin = abs(_safe_float(winner["points"]) - _safe_float(loser["points"]))

            matchup_rows.append(
                {
                    "week": _safe_int(winner["week"]),
                    "winner": str(winner["owner"]),
                    "loser": str(loser["owner"]),
                    "margin": margin,
                }
            )

        if matchup_rows:
            closest = min(matchup_rows, key=lambda x: x["margin"])
            blowout = max(matchup_rows, key=lambda x: x["margin"])

            summary["closest_matchup"] = f"Week {closest['week']}: {closest['winner']} over {closest['loser']}"
            summary["closest_margin"] = _safe_float(closest["margin"])

            summary["biggest_blowout"] = f"Week {blowout['week']}: {blowout['winner']} over {blowout['loser']}"
            summary["biggest_blowout_margin"] = _safe_float(blowout["margin"])

    return summary


def _history_chart(df_weekly: pd.DataFrame) -> str:
    chart_df = _filtered_season_df(df_weekly)
    if chart_df.empty or not {"week", "owner", "points"}.issubset(chart_df.columns):
        return "<div class='history-empty'>No weekly scoring data available for this season.</div>"

    fig = go.Figure()

    for owner, grp in chart_df.groupby("owner"):
        grp = grp.sort_values("week")
        fig.add_trace(
            go.Scatter(
                x=grp["week"],
                y=grp["points"],
                mode="lines+markers",
                name=str(owner),
                hovertemplate="%{fullData.name}<br>Week %{x}<br>%{y:.1f} pts<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=430,
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0),
        xaxis=dict(title="Week", dtick=1),
        yaxis=dict(title="Points"),
    )

    return plotly_plot(
        fig,
        include_plotlyjs=False,
        output_type="div",
        config={"displayModeBar": False},
    )


def _summary_card(label: str, value: str, sub: str = "", featured: bool = False, card_type: str = "") -> str:
    cls = "history-card"
    moment_attr = ""
    if card_type == "champion":
        cls += " is-featured is-champion br-champ"
        # Crowning moment: the trophy label draws in, the team name rises, and a
        # burst of gold confetti fires when the card first scrolls into view.
        moment_attr = ' data-br-moment="champion" data-br-confetti="gold" data-br-confetti-delay="650"'
    elif card_type == "runner_up":
        cls += " is-featured is-runner-up"
    elif featured:
        cls += " is-featured"
    return f"""
    <div class="{cls}"{moment_attr}>
      <div class="history-card-label">{label}</div>
      <div class="history-card-value">{value}</div>
      {f'<div class="history-card-sub">{sub}</div>' if sub else ''}
    </div>
    """


def _standings_table(team_stats: pd.DataFrame) -> str:
    df = sort_team_stats(team_stats)
    if df.empty:
        return """
        <div class="history-section-card">
          <div class="history-empty">No final standings found for this season.</div>
        </div>
        """

    rows = []
    for _, row in df.iterrows():
        rows.append(
            f"""
            <tr>
              <td>{_safe_int(row.get('Rank'))}</td>
              <td>{row.get('owner', '-')}</td>
              <td>{_record_str(row)}</td>
              <td>{_safe_float(row.get('PF')):.1f}</td>
              <td>{_safe_float(row.get('PA')):.1f}</td>
              <td>{_safe_float(row.get('AVG')):.1f}</td>
              <td>{_safe_float(row.get('STD')):.1f}</td>
            </tr>
            """
        )

    return f"""
    <div class="history-section-card">
      <div class="history-table-wrap">
        <table class="history-table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Team</th>
              <th>Record</th>
              <th>PF</th>
              <th>PA</th>
              <th>AVG</th>
              <th>STD</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
      </div>
    </div>
    """


def get_history_summary_html(history_ctx: dict) -> str:
    """Generate season awards/summary section HTML."""
    summary = _build_summary(history_ctx)

    featured_cards_html = "".join(
        [
            _summary_card(
                "<i class='fa-solid fa-trophy' aria-hidden='true'></i> Champion",
                summary["champion"],
                f"Regular season record: {summary['champion_record']}",
                card_type="champion",
            ),
            _summary_card(
                "<i class='fa-solid fa-medal' aria-hidden='true'></i> Runner-Up",
                summary["runner_up"],
                f"Regular season record: {summary['runner_up_record']}",
                card_type="runner_up",
            ),
            _summary_card(
                "Scoring Leader",
                summary["top_scorer_team"],
                f"{summary['top_scorer_value']:.1f} PF • {summary['top_scorer_avg']:.1f} avg pts",
                featured=True,
            ),
        ]
    )

    compact_cards_html = "".join(
        [
            _summary_card(
                "Best Defense",
                summary["best_defense_team"],
                f"{summary['best_defense_value']:.1f} PA",
            ),
            _summary_card(
                "Highest Week",
                summary["highest_week_team"],
                f"{summary['highest_week_value']:.1f} pts",
            ),
            _summary_card(
                "Lowest Week",
                summary["lowest_week_team"],
                f"{summary['lowest_week_value']:.1f} pts",
            ),
            _summary_card(
                "Closest Matchup",
                summary["closest_matchup"],
                f"{summary['closest_margin']:.1f} point margin",
            ),
            _summary_card(
                "Biggest Blowout",
                summary["biggest_blowout"],
                f"{summary['biggest_blowout_margin']:.1f} point margin",
            ),
            _summary_card(
                "Unluckiest Team",
                summary["unluckiest_team"],
                (
                    f"{summary['unluckiest_delta']} spots below PF rank"
                    if summary["unluckiest_delta"] > 0
                    else "Matched or beat its scoring rank"
                ),
            ),
        ]
    )

    return f"""
    <div class="history-awards-grid">
      {featured_cards_html}
      {compact_cards_html}
    </div>
    """


def get_history_standings_html(history_ctx: dict) -> str:
    """Generate standings table HTML."""
    league = history_ctx.get("league") or {}
    df_weekly = history_ctx.get("df_weekly", pd.DataFrame())

    regular_season_team_stats = build_regular_season_team_stats(df_weekly, league)
    return _standings_table(regular_season_team_stats)


def _build_rivalry_card(
        history_ctx: dict,
        base_platform: str,
        base_season: int,
        base_league_id: str,
) -> str:
    """All-time head-to-head card: pick two managers, see their full rivalry."""
    users = history_ctx.get("users") or []
    opts = []
    for u in users:
        uid = str(u.get("user_id") or "").strip()
        if not uid:
            continue
        metadata = u.get("metadata") or {}
        name = (
            u.get("display_name")
            or metadata.get("team_name")
            or u.get("username")
            or uid
        )
        opts.append((str(name), uid))
    if len(opts) < 2:
        return ""
    opts.sort(key=lambda t: t[0].lower())
    options_html = "".join(
        f"<option value='{_esc(uid)}'>{_esc(name)}</option>" for name, uid in opts
    )

    card = """
      <div class="card rivalry-card">
        <div class="card-header">
          <h2>Rivalry Tracker</h2>
          <div style="font-size:13px;color:var(--text-muted);">All-time head-to-head, every season included</div>
        </div>
        <div class="card-body">
          <div class="rivalry-controls">
            <select id="rivalrySelA" class="rivalry-select">
              <option value="">Select manager…</option>
              __OPTIONS__
            </select>
            <span class="rivalry-vs">VS</span>
            <select id="rivalrySelB" class="rivalry-select">
              <option value="">Select manager…</option>
              __OPTIONS__
            </select>
            <button id="rivalryGoBtn" class="rivalry-go-btn" disabled>Compare</button>
          </div>
          <div id="rivalryResult" class="rivalry-result"></div>
        </div>
      </div>
      <script>
      (function() {
        var selA = document.getElementById('rivalrySelA');
        var selB = document.getElementById('rivalrySelB');
        var goBtn = document.getElementById('rivalryGoBtn');
        var result = document.getElementById('rivalryResult');
        if (!selA || !selB || !goBtn) return;

        function syncBtn() {
          goBtn.disabled = !(selA.value && selB.value && selA.value !== selB.value);
        }
        selA.addEventListener('change', syncBtn);
        selB.addEventListener('change', syncBtn);

        function nameOf(sel) { return sel.options[sel.selectedIndex].text; }

        goBtn.addEventListener('click', function() {
          var a = selA.value, b = selB.value;
          if (!a || !b || a === b) return;
          result.innerHTML = '<div style="text-align:center;padding:24px;color:var(--text-muted);">Loading rivalry…</div>';
          fetch('/api/rivalry/__PLATFORM__/__SEASON__/__LEAGUE_ID__?a=' + encodeURIComponent(a) + '&b=' + encodeURIComponent(b))
            .then(function(r) { return r.json(); })
            .then(function(d) {
              if (d.error) { result.innerHTML = '<div class="rivalry-empty">' + d.error + '</div>'; return; }
              var games = d.games || [];
              if (!games.length) {
                result.innerHTML = '<div class="rivalry-empty">These two managers have never faced each other.</div>';
                return;
              }
              var nA = nameOf(selA), nB = nameOf(selB);
              var margins = games.map(function(g) { return Math.abs(g.a_pts - g.b_pts); });
              var avgMargin = margins.reduce(function(s, m) { return s + m; }, 0) / games.length;
              var blowout = games.reduce(function(best, g) {
                return Math.abs(g.a_pts - g.b_pts) > Math.abs(best.a_pts - best.b_pts) ? g : best;
              }, games[0]);
              var bWinner = blowout.a_pts > blowout.b_pts ? nA : nB;
              var bMargin = Math.abs(blowout.a_pts - blowout.b_pts);

              var streakLen = 0, streakSide = null;
              for (var i = games.length - 1; i >= 0; i--) {
                var w = games[i].a_pts > games[i].b_pts ? 'a' : (games[i].b_pts > games[i].a_pts ? 'b' : null);
                if (w === null) break;
                if (streakSide === null) { streakSide = w; streakLen = 1; }
                else if (w === streakSide) { streakLen++; }
                else break;
              }
              var streakName = streakSide === 'a' ? nA : nB;

              var aLead = d.wins_a > d.wins_b, bLead = d.wins_b > d.wins_a;
              var total = d.wins_a + d.wins_b + (d.ties || 0);
              var barPct = total > 0 ? Math.round((d.wins_a / total) * 100) : 50;

              // Score banner
              var html = '<div class="rivalry-banner">'
                + '<div class="rivalry-side' + (aLead ? ' rivalry-side--leader' : '') + '">'
                +   '<div class="rivalry-name">' + nA + '</div>'
                +   '<div class="rivalry-wins' + (aLead ? ' rivalry-wins--lead' : '') + '">' + d.wins_a + '</div>'
                + '</div>'
                + '<div class="rivalry-divider">'
                +   '<div class="rivalry-divider-label">vs</div>'
                +   '<div class="rivalry-divider-dash">–</div>'
                + '</div>'
                + '<div class="rivalry-side' + (bLead ? ' rivalry-side--leader' : '') + '">'
                +   '<div class="rivalry-name">' + nB + '</div>'
                +   '<div class="rivalry-wins' + (bLead ? ' rivalry-wins--lead' : '') + '">' + d.wins_b + '</div>'
                + '</div>'
                + '</div>';

              if (d.ties) html += '<div class="rivalry-ties">' + d.ties + ' tie' + (d.ties > 1 ? 's' : '') + '</div>';

              // Win-rate bar
              html += '<div class="rivalry-bar-wrap">'
                + '<span class="rivalry-bar-label">' + barPct + '%</span>'
                + '<div class="rivalry-bar-track"><div class="rivalry-bar-fill" style="width:' + barPct + '%"></div></div>'
                + '<span class="rivalry-bar-label rivalry-bar-label--right">' + (100 - barPct) + '%</span>'
                + '</div>';

              // Stat chips
              html += '<div class="rivalry-chips">'
                + '<span class="rivalry-chip"><i class="fa-solid fa-calendar rivalry-chip-icon"></i>' + games.length + ' meetings</span>'
                + '<span class="rivalry-chip"><i class="fa-solid fa-chart-simple rivalry-chip-icon"></i>Total pts: ' + d.pts_a.toFixed(1) + ' – ' + d.pts_b.toFixed(1) + '</span>'
                + '<span class="rivalry-chip"><i class="fa-solid fa-medal rivalry-chip-icon"></i>Avg margin: ' + avgMargin.toFixed(1) + '</span>'
                + (streakLen > 1 ? '<span class="rivalry-chip rivalry-chip-streak"><i class="fa-solid fa-fire"></i>' + streakName + ' won ' + streakLen + ' straight</span>' : '')
                + '<span class="rivalry-chip rivalry-chip-blowout"><i class="fa-solid fa-bolt"></i>' + bWinner + ' by ' + bMargin.toFixed(1) + ' (' + blowout.season + ' wk ' + blowout.week + ')</span>'
                + '</div>';

              // Matchup history table
              html += '<div class="rivalry-table-wrap"><table class="rivalry-table"><thead><tr>'
                + '<th>Season</th><th>Week</th>'
                + '<th style="text-align:right">' + nA + '</th>'
                + '<th style="text-align:right">' + nB + '</th>'
                + '</tr></thead><tbody>';
              var recent = games.slice(-12).reverse();
              recent.forEach(function(g) {
                var aWin = g.a_pts > g.b_pts, bWin = g.b_pts > g.a_pts;
                html += '<tr><td style="color:var(--text-muted)">' + g.season + '</td><td style="color:var(--text-muted)">Wk ' + g.week + '</td>'
                  + '<td style="text-align:right" class="' + (aWin ? 'rivalry-w' : '') + '">' + g.a_pts.toFixed(1) + '</td>'
                  + '<td style="text-align:right" class="' + (bWin ? 'rivalry-w' : '') + '">' + g.b_pts.toFixed(1) + '</td></tr>';
              });
              html += '</tbody></table></div>';
              if (games.length > 12) html += '<div class="rivalry-table-footer">Showing the 12 most recent of ' + games.length + ' meetings</div>';
              result.innerHTML = html;
            })
            .catch(function() {
              result.innerHTML = '<div class="rivalry-empty">Could not load rivalry data. Try again.</div>';
            });
        });
      })();
      </script>
    """
    return (
        card
        .replace("__OPTIONS__", options_html)
        .replace("__PLATFORM__", _esc(str(base_platform)))
        .replace("__SEASON__", _esc(str(base_season)))
        .replace("__LEAGUE_ID__", _esc(str(base_league_id)))
    )


def _wrapped_longest_win_streak(df_weekly: pd.DataFrame, league: dict) -> tuple:
    """(owner, length) of the season's longest single-team win streak, or
    ('', 0). Regular season, finalized games only."""
    try:
        df = _filtered_season_df(df_weekly)
        if df is None or df.empty or "matchup_id" not in df.columns:
            return "", 0
        df = df.copy()
        df["week"] = pd.to_numeric(df["week"], errors="coerce")
        results: dict[str, list] = {}
        for (_wk, _mid), grp in df.groupby(["week", "matchup_id"]):
            if len(grp) != 2:
                continue
            g = grp.sort_values("owner")
            a, b = g.iloc[0], g.iloc[1]
            pa, pb = float(a.get("points", 0) or 0), float(b.get("points", 0) or 0)
            if pa == pb:
                continue
            win, lose = (a, b) if pa > pb else (b, a)
            results.setdefault(str(win["owner"]), []).append((int(win["week"]), "W"))
            results.setdefault(str(lose["owner"]), []).append((int(lose["week"]), "L"))
        best_owner, best_len = "", 0
        for owner, seq in results.items():
            run = 0
            for _wk, r in sorted(seq):
                run = run + 1 if r == "W" else 0
                if run > best_len:
                    best_len, best_owner = run, owner
        return best_owner, best_len
    except Exception:
        return "", 0


def _build_wrapped_slides(summary: dict, df_weekly: pd.DataFrame, league: dict,
                          league_name: str, season) -> list:
    """Ordered 'Season Wrapped' story slides built from the season summary.
    Each slide: {kind, eyebrow, big, num, dp, suffix, label, sub}."""
    def _txt(kind, eyebrow, big, label, sub):
        return {"kind": kind, "eyebrow": eyebrow, "big": big, "num": False,
                "dp": 0, "suffix": "", "label": label, "sub": sub}

    def _num(kind, eyebrow, val, dp, suffix, label, sub):
        return {"kind": kind, "eyebrow": eyebrow, "big": f"{val:.{dp}f}", "num": True,
                "dp": dp, "suffix": suffix, "label": label, "sub": sub}

    slides = [_txt("intro", f"{season} SEASON", league_name, "Wrapped",
                   "A look back at the year that was")]

    if summary.get("top_scorer_value"):
        slides.append(_num("topscore", "MOST POINTS ON THE YEAR",
                           summary["top_scorer_value"], 1, "",
                           summary.get("top_scorer_team", "-"),
                           f"{summary.get('top_scorer_avg', 0):.1f} avg per week — the league's top scoring machine"))

    if summary.get("highest_week_value"):
        slides.append(_num("highweek", "BIGGEST SINGLE WEEK",
                           summary["highest_week_value"], 1, "",
                           summary.get("highest_week_team", "-"),
                           "The highest one-week score anyone put up all season"))

    owner, streak_len = _wrapped_longest_win_streak(df_weekly, league)
    if streak_len >= 3:
        slides.append(_num("streak", "HOTTEST STREAK", float(streak_len), 0, " straight",
                           owner, "The longest win streak of the season"))

    if summary.get("biggest_blowout_margin"):
        slides.append(_num("blowout", "BIGGEST BLOWOUT",
                           summary["biggest_blowout_margin"], 1, " pts",
                           summary.get("biggest_blowout", "-"),
                           "The most lopsided result of the year"))

    if summary.get("closest_margin"):
        slides.append(_num("nailbiter", "CLOSEST GAME",
                           summary["closest_margin"], 1, " pts",
                           summary.get("closest_matchup", "-"),
                           "Decided by the slimmest margin all season"))

    if summary.get("runner_up") and summary.get("runner_up") not in ("-", None):
        slides.append(_txt("runnerup", "RUNNER-UP", summary["runner_up"], "So close",
                           f"Finished {summary.get('runner_up_record', '')} — one game short"))

    if summary.get("champion") and summary.get("champion") not in ("-", None):
        slides.append(_txt("champion", f"{season} CHAMPION", summary["champion"],
                           "", f"{summary.get('champion_record', '')} — league champion"))

    return slides


def _render_season_wrapped(slides: list, league_name: str, season) -> tuple:
    """Return (launch_button_html, overlay_html) for the Season Wrapped stories
    experience. Returns ('', '') when there isn't enough to tell a story."""
    if len(slides) < 3:
        return "", ""

    bars = "".join("<span class='wrapped-bar'><i></i></span>" for _ in slides)

    slide_html = []
    for s in slides:
        if s["num"]:
            big = (f"<div class='wrapped-big' data-w-count='{s['big']}' "
                   f"data-w-dp='{s['dp']}' data-w-suffix=\"{_esc(s['suffix'], quote=True)}\">0</div>")
        else:
            big = f"<div class='wrapped-big wrapped-big-text'>{_esc(str(s['big']))}</div>"
        slide_html.append(
            f"<section class='wrapped-slide' data-kind='{s['kind']}'>"
            f"<div class='wrapped-eyebrow'>{_esc(str(s['eyebrow']))}</div>"
            f"{big}"
            + (f"<div class='wrapped-label'>{_esc(str(s['label']))}</div>" if s['label'] else "")
            + f"<div class='wrapped-sub'>{_esc(str(s['sub']))}</div>"
            f"</section>"
        )

    button = (
        "<button type='button' class='wrapped-launch' id='wrappedLaunch'>"
        "<i class='fa-solid fa-wand-magic-sparkles'></i> Season Wrapped</button>"
    )

    overlay = f"""
    <div class="wrapped-overlay" id="wrappedOverlay" hidden aria-hidden="true">
      <div class="wrapped-progress">{bars}</div>
      <button type="button" class="wrapped-close" id="wrappedClose" aria-label="Close">&times;</button>
      <div class="wrapped-stage" id="wrappedStage">{''.join(slide_html)}</div>
      <button type="button" class="wrapped-tap wrapped-tap-prev" id="wrappedPrev" aria-label="Previous"></button>
      <button type="button" class="wrapped-tap wrapped-tap-next" id="wrappedNext" aria-label="Next"></button>
      <div class="wrapped-hint">Tap to advance · Esc to close</div>
    </div>
    <script>{_WRAPPED_JS}</script>
    """
    return button, overlay


_WRAPPED_JS = r"""
(function () {
  var launch = document.getElementById('wrappedLaunch');
  var overlay = document.getElementById('wrappedOverlay');
  if (!launch || !overlay || overlay.__wrapBound) return;
  overlay.__wrapBound = true;
  var stage = document.getElementById('wrappedStage');
  var slides = Array.prototype.slice.call(stage.querySelectorAll('.wrapped-slide'));
  var bars = Array.prototype.slice.call(overlay.querySelectorAll('.wrapped-bar'));
  var idx = 0;

  function playSlide(n) {
    slides.forEach(function (s, i) {
      s.classList.toggle('active', i === n);
      if (i < n) s.classList.add('seen'); else s.classList.remove('seen');
    });
    bars.forEach(function (b, i) { b.classList.toggle('filled', i < n); b.classList.toggle('current', i === n); });
    var el = slides[n];
    if (!el) return;
    // Count-up the big number when the slide arrives.
    var num = el.querySelector('[data-w-count]');
    if (num && window.brCountUp) {
      window.brCountUp(num, { to: parseFloat(num.getAttribute('data-w-count')),
        dp: parseInt(num.getAttribute('data-w-dp') || '0', 10),
        suffix: num.getAttribute('data-w-suffix') || '', dur: 1100 });
    }
    if (el.getAttribute('data-kind') === 'champion' && window.brConfetti) {
      setTimeout(function () { window.brConfetti(el, { palette: ['#f5c451','#e0a828','#fff1c2','#ffffff'], y: el.clientHeight * 0.4, count: 120 }); }, 350);
    }
  }
  function go(n) {
    if (n < 0) return;
    if (n >= slides.length) { close(); return; }
    idx = n; playSlide(idx);
  }
  function open() {
    overlay.hidden = false; overlay.setAttribute('aria-hidden', 'false');
    document.documentElement.style.overflow = 'hidden';
    idx = 0; requestAnimationFrame(function () { playSlide(0); });
  }
  function close() {
    overlay.hidden = true; overlay.setAttribute('aria-hidden', 'true');
    document.documentElement.style.overflow = '';
  }

  launch.addEventListener('click', open);
  document.getElementById('wrappedClose').addEventListener('click', close);
  document.getElementById('wrappedNext').addEventListener('click', function () { go(idx + 1); });
  document.getElementById('wrappedPrev').addEventListener('click', function () { go(idx - 1); });
  document.addEventListener('keydown', function (e) {
    if (overlay.hidden) return;
    if (e.key === 'Escape') close();
    else if (e.key === 'ArrowRight' || e.key === ' ') { e.preventDefault(); go(idx + 1); }
    else if (e.key === 'ArrowLeft') go(idx - 1);
  });
  // Swipe on touch.
  var sx = null;
  stage.addEventListener('touchstart', function (e) { sx = e.touches[0].clientX; }, { passive: true });
  stage.addEventListener('touchend', function (e) {
    if (sx === null) return;
    var dx = e.changedTouches[0].clientX - sx; sx = null;
    if (Math.abs(dx) > 40) go(idx + (dx < 0 ? 1 : -1));
  }, { passive: true });
})();
"""


def build_history_body(
        history_ctx: dict,
        available_seasons: List[int],
        base_platform: str,
        base_season: int,
        base_league_id: str,
        selected_history_season: int,
        resolved_history_league_id: str,
        prerendered: Dict[str, str] | None = None,
) -> str:
    league = history_ctx.get("league") or {}
    df_weekly = history_ctx.get("df_weekly", pd.DataFrame())

    summary = _build_summary(history_ctx)

    # Add summary to history_ctx for AI generation
    history_ctx["summary"] = summary

    # Use AI to generate league season summary
    recap_line = get_league_season_summary(history_ctx, selected_history_season)

    options_html = []
    for yr in available_seasons:
        href = url_for(
            "page_history",
            platform=base_platform,
            season=base_season,
            league_id=base_league_id,
            history_season=yr,
        )
        selected = "selected" if yr == selected_history_season else ""
        options_html.append(f"<option value='{href}' {selected}>{yr}</option>")

    league_name = league.get("name") or "League History"

    # Season Wrapped: a stories-style recap, available once there's a champion.
    _wrapped_slides = _build_wrapped_slides(
        summary, df_weekly, league, league_name, selected_history_season
    )
    _wrapped_btn, _wrapped_overlay = _render_season_wrapped(
        _wrapped_slides, league_name, selected_history_season
    )

    # Loading spinner HTML (unused when prerendered sections are provided)
    loading_spinner = """
    <div class="history-loading-state">
      <div class="loading-spinner" style="margin: 20px auto; width: 30px; height: 30px; border: 3px solid var(--border); border-radius: 50%; border-top-color: var(--accent); animation: spin 1s linear infinite; border-right-color: transparent;"></div>
      <div style="text-align: center; color: var(--text-subtle); font-size: 13px; margin-top: 12px;">Loading...</div>
    </div>
    """
    awards_html    = prerendered["summary"]   if prerendered else loading_spinner
    standings_html = prerendered["standings"] if prerendered else loading_spinner
    chart_html     = prerendered["chart"]     if prerendered else loading_spinner
    tour_input     = '<input type="hidden" id="historyTourMode" value="1">' if prerendered else ""

    rivalry_html = _build_rivalry_card(
        history_ctx, base_platform, base_season, base_league_id,
    )

    return f"""
    <div class="history-page">
      <!-- Hidden inputs for JavaScript -->
      <input type="hidden" id="leagueIdInput" value="{_esc(str(base_league_id))}">
      <input type="hidden" id="seasonInput" value="{_esc(str(base_season))}">
      <input type="hidden" id="platformInput" value="{_esc(str(base_platform))}">
      <input type="hidden" id="historySeasonInput" value="{_esc(str(selected_history_season))}">
      <input type="hidden" id="resolvedLeagueIdInput" value="{_esc(str(resolved_history_league_id))}">
      {tour_input}

      <div class="history-header">
        <div>
          <div class="history-kicker">League History</div>
          <h1 class="history-title">
            <span class="history-title-accent">{_esc(league_name)}</span> • {selected_history_season}
          </h1>
          <p class="history-subtitle">{_esc(recap_line)}</p>
        </div>

        <div style="display:flex;flex-direction:column;align-items:flex-end;gap:12px;">
          {_wrapped_btn}
          <a href="/{base_platform}/{base_season}/{base_league_id}/awards" class="awards-page-nav-link">
            <i class="fa-solid fa-trophy"></i>
            All-Time Awards
          </a>
          <div class="history-season-picker">
            <label for="history-season-select">Season</label>
            <select
              id="history-season-select"
              onchange="if (this.value) window.location.href = this.value;"
            >
              {''.join(options_html)}
            </select>
          </div>
        </div>
      </div>

      <div class="history-top-grid">
        <div class="card history-awards-panel">
          <div class="card-header"><h2>Season Awards</h2></div>
          <div class="card-body" id="historyAwardsContent">
            {awards_html}
          </div>
        </div>

        <div class="card history-standings-panel">
          <div class="card-header"><h2>Regular Season Standings</h2></div>
          <div class="card-body" style="padding-top:0;" id="historyStandingsContent">
            {standings_html}
          </div>
        </div>
      </div>

      <div class="history-top-grid">
        <div class="card history-chart-panel">
          <div class="card-header"><h2>Season Trend</h2></div>
          <div class="card-body" id="historyChartContent">
            {chart_html}
          </div>
        </div>

        <div class="card history-recap-panel">
          <div class="card-header">
            <h2>Season Recap</h2>
            <div class="history-recap-controls">
              <select id="recapTeamDropdown" class="recap-team-dropdown">
                <option value="">Select your team...</option>
              </select>
              <button id="generateRecapBtn" class="recap-generate-btn" disabled>
                Generate AI Recap
              </button>
            </div>
          </div>
          <div class="card-body history-recap-content">
            <div class="otc-ai-empty" id="aiLoadingState" style="display:none;">
              <div class="otc-ai-empty-title">Analyzing Season...</div>
              <div class="otc-ai-empty-sub">
                <div class="loading-spinner" style="margin: 10px auto; width: 30px; height: 30px; border: 3px solid var(--border); border-radius: 50%; border-top-color: var(--accent); animation: spin 1s linear infinite; border-right-color: transparent;"></div>
              </div>
            </div>
            <div id="aiAnalysisResult" class="recap-result" style="display:none;"></div>
            <div id="aiEmptyState" class="recap-empty">
              <div class="recap-empty-title">AI Season Recap</div>
              <div class="recap-empty-sub">
                Select your team above to generate a personalized season recap with AI analysis.
              </div>
            </div>
          </div>
        </div>
      </div>

      {rivalry_html}
    </div>
    {_wrapped_overlay}
    """
