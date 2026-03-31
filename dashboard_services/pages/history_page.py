from __future__ import annotations

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from flask import url_for
from plotly.offline import plot as plotly_plot
from typing import Any, Dict, List

from dashboard_services.ai.history_recap import get_league_season_summary
from dashboard_services.platform_api import get_bracket


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
    if team_stats is None or team_stats.empty or not team_name or team_name == "—":
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
        return "—", "—"

    # Only consider completed matchups
    completed = [
        m for m in winners_bracket
        if m.get("w") is not None and m.get("l") is not None
    ]
    if not completed:
        return "—", "—"

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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
        return "—"
    return (
            roster_map.get(str(roster_id))
            or roster_map.get(roster_id)
            or f"Roster {roster_id}"
    )


def _title_game_names(ctx: dict) -> tuple[str, str]:
    platform = ctx["platform"]
    season = int(ctx["season"])
    league_id = ctx.get("resolved_league_id") or ctx["league_id"]
    roster_map = ctx.get("roster_map") or {}

    try:
        winners_bracket = get_bracket(platform, league_id, season, "winners") or []
    except Exception:
        winners_bracket = []

    if not winners_bracket:
        return "—", "—"

    # Prefer latest round / latest matchup
    finalists = sorted(
        winners_bracket,
        key=lambda m: (_safe_int(m.get("r"), 0), _safe_int(m.get("m"), 0)),
        reverse=True,
    )

    for matchup in finalists:
        winner_id = matchup.get("w")
        loser_id = matchup.get("l")
        if winner_id is not None and loser_id is not None:
            return (
                _team_name_from_roster_id(roster_map, winner_id),
                _team_name_from_roster_id(roster_map, loser_id),
            )

    return "—", "—"


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
        "champion_record": _record_str(pd.Series(champion_stats)) if champion_stats else "—",
        "runner_up": runner_up,
        "runner_up_record": _record_str(pd.Series(runner_up_stats)) if runner_up_stats else "—",
        "top_scorer_team": "—",
        "top_scorer_value": 0.0,
        "top_scorer_avg": 0.0,
        "best_defense_team": "—",
        "best_defense_value": 0.0,
        "highest_week_team": "—",
        "highest_week_value": 0.0,
        "lowest_week_team": "—",
        "lowest_week_value": 0.0,
        "closest_matchup": "—",
        "closest_margin": 0.0,
        "biggest_blowout": "—",
        "biggest_blowout_margin": 0.0,
        "unluckiest_team": "—",
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

        summary["highest_week_team"] = str(hi.get("owner", "—"))
        summary["highest_week_value"] = _safe_float(hi.get("points"))

        summary["lowest_week_team"] = str(lo.get("owner", "—"))
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


def _build_recap_line(summary: dict, season: int) -> str:
    champ = summary["champion"]
    runner = summary["runner_up"]
    scoring_leader = summary["top_scorer_team"]
    unlucky = summary["unluckiest_team"]

    parts = []
    if champ != "—" and runner != "—":
        parts.append(f"{champ} won the {season} title over {runner}.")
    elif champ != "—":
        parts.append(f"{champ} finished as the {season} champion.")

    if scoring_leader != "—":
        parts.append(f"{scoring_leader} led the league in total points.")

    if unlucky != "—" and summary["unluckiest_delta"] > 0:
        parts.append(
            f"{unlucky} was the rough-luck team, finishing {summary['unluckiest_delta']} spots below its PF rank."
        )

    return " ".join(parts) or f"Review the biggest outcomes and trends from the {season} season."


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


def _summary_card(label: str, value: str, sub: str = "", featured: bool = False) -> str:
    featured_cls = " is-featured" if featured else ""
    return f"""
    <div class="history-card{featured_cls}">
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
              <td>{row.get('owner', '—')}</td>
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
                "Champion",
                summary["champion"],
                f"Regular season record: {summary['champion_record']}",
                featured=True,
            ),
            _summary_card(
                "Runner-Up",
                summary["runner_up"],
                f"Regular season record: {summary['runner_up_record']}",
                featured=True,
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


def get_history_chart_html(history_ctx: dict) -> str:
    """Generate season trend chart HTML."""
    df_weekly = history_ctx.get("df_weekly", pd.DataFrame())
    return _history_chart(df_weekly)


def build_history_body(
        history_ctx: dict,
        available_seasons: List[int],
        base_platform: str,
        base_season: int,
        base_league_id: str,
        selected_history_season: int,
        resolved_history_league_id: str,
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

    # Loading spinner HTML
    loading_spinner = """
    <div class="history-loading-state">
      <div class="loading-spinner" style="margin: 20px auto; width: 30px; height: 30px; border: 3px solid #f3f4f6; border-radius: 50%; border-top-color: #3498db; animation: spin 1s linear infinite; border-right-color: transparent;"></div>
      <div style="text-align: center; color: #94a3b8; font-size: 13px; margin-top: 12px;">Loading...</div>
    </div>
    """

    return f"""
    <div class="history-page">
      <!-- Hidden inputs for JavaScript -->
      <input type="hidden" id="leagueIdInput" value="{base_league_id}">
      <input type="hidden" id="seasonInput" value="{base_season}">
      <input type="hidden" id="platformInput" value="{base_platform}">
      <input type="hidden" id="historySeasonInput" value="{selected_history_season}">
      <input type="hidden" id="resolvedLeagueIdInput" value="{resolved_history_league_id}">

      <div class="history-header">
        <div>
          <div class="history-kicker">League History</div>
          <h1 class="history-title">
            <span class="history-title-accent">{league_name}</span> • {selected_history_season}
          </h1>
          <p class="history-subtitle">{recap_line}</p>
        </div>

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

      <div class="history-top-grid">
        <div class="history-section-card history-awards-panel">
          <div class="history-section-title">Season Awards</div>
          <div id="historyAwardsContent">
            {loading_spinner}
          </div>
        </div>

        <div class="history-section-card history-standings-panel">
          <div class="history-section-title">Regular Season Standings</div>
          <div id="historyStandingsContent">
            {loading_spinner}
          </div>
        </div>
      </div>

      <div class="history-top-grid">
        <div class="history-section-card history-chart-panel">
          <div class="history-section-title">Season Trend</div>
          <div id="historyChartContent">
            {loading_spinner}
          </div>
        </div>

        <div class="history-section-card history-recap-panel">
          <div class="history-recap-header">
            <div class="history-section-title">Season Recap</div>
            <div class="history-recap-controls">
              <select id="recapTeamDropdown" class="recap-team-dropdown">
                <option value="">Select your team...</option>
              </select>
              <button id="generateRecapBtn" class="recap-generate-btn" disabled>
                Generate AI Recap
              </button>
            </div>
          </div>
          <div class="history-recap-content">
            <div class="otc-ai-empty" id="aiLoadingState" style="display:none;">
              <div class="otc-ai-empty-title">Analyzing Season...</div>
              <div class="otc-ai-empty-sub">
                <div class="loading-spinner" style="margin: 10px auto; width: 30px; height: 30px; border: 3px solid #f3f4f6; border-radius: 50%; border-top-color: #3498db; animation: spin 1s linear infinite; border-right-color: transparent;"></div>
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
    </div>
    """
