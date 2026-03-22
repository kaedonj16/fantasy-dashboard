from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from flask import url_for
from plotly.offline import plot as plotly_plot

from dashboard_services.platform_api import get_bracket


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


def _sort_team_stats(team_stats: pd.DataFrame) -> pd.DataFrame:
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


def _build_summary(ctx: dict) -> dict:
    df_weekly = _filtered_season_df(ctx.get("df_weekly", pd.DataFrame()))
    team_stats = _sort_team_stats(ctx.get("team_stats", pd.DataFrame()))

    champion, runner_up = _title_game_names(ctx)

    summary = {
        "champion": champion,
        "runner_up": runner_up,
        "top_scorer_team": "—",
        "top_scorer_value": 0.0,
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

    if not df_weekly.empty and "owner" in df_weekly.columns and "points" in df_weekly.columns:
        hi = df_weekly.loc[df_weekly["points"].idxmax()]
        lo = df_weekly.loc[df_weekly["points"].idxmin()]

        summary["highest_week_team"] = str(hi.get("owner", "—"))
        summary["highest_week_value"] = _safe_float(hi.get("points"))

        summary["lowest_week_team"] = str(lo.get("owner", "—"))
        summary["lowest_week_value"] = _safe_float(lo.get("points"))

    if (
        not df_weekly.empty
        and {"week", "matchup_id", "owner", "points"}.issubset(df_weekly.columns)
    ):
        matchup_rows = []
        for (_, matchup_id), grp in df_weekly.groupby(["week", "matchup_id"]):
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

            summary["closest_matchup"] = (
                f"Week {closest['week']}: {closest['winner']} over {closest['loser']}"
            )
            summary["closest_margin"] = _safe_float(closest["margin"])

            summary["biggest_blowout"] = (
                f"Week {blowout['week']}: {blowout['winner']} over {blowout['loser']}"
            )
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


def _summary_card(label: str, value: str, sub: str = "") -> str:
    return f"""
    <div class="history-card">
      <div class="history-card-label">{label}</div>
      <div class="history-card-value">{value}</div>
      {f'<div class="history-card-sub">{sub}</div>' if sub else ''}
    </div>
    """


def _standings_table(team_stats: pd.DataFrame) -> str:
    df = _sort_team_stats(team_stats)
    if df.empty:
        return """
        <div class="history-section-card">
          <div class="history-section-title">Final Standings</div>
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
      <div class="history-section-title">Final Standings</div>
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


def build_history_body(ctx: dict, available_seasons: List[int]) -> str:
    league = ctx.get("league") or {}
    platform = ctx["platform"]
    season = int(ctx["season"])
    league_id = ctx["league_id"]
    df_weekly = ctx.get("df_weekly", pd.DataFrame())
    team_stats = ctx.get("team_stats", pd.DataFrame())

    summary = _build_summary(ctx)
    recap_line = _build_recap_line(summary, season)
    chart_html = _history_chart(df_weekly)

    options_html = []
    for yr in available_seasons:
        href = url_for(
            "page_history",
            platform=platform,
            season=yr,
            league_id=league_id,
            explicit=1,
        )
        selected = "selected" if yr == season else ""
        options_html.append(f"<option value='{href}' {selected}>{yr}</option>")

    league_name = league.get("name") or "League History"

    cards_html = "".join(
        [
            _summary_card("Champion", summary["champion"]),
            _summary_card("Runner-Up", summary["runner_up"]),
            _summary_card(
                "Scoring Leader",
                summary["top_scorer_team"],
                f"{summary['top_scorer_value']:.1f} PF",
            ),
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

    standings_html = _standings_table(team_stats)

    return f"""
    <div class="history-page">
      <div class="history-header">
        <div>
          <div class="history-kicker">League History</div>
          <h1 class="history-title">{league_name} • {season}</h1>
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

      <div class="history-summary-grid">
        {cards_html}
      </div>

      <div class="history-section-card">
        <div class="history-section-title">Season Trend</div>
        <div class="history-chart-wrap">
          {chart_html}
        </div>
      </div>

      {standings_html}
    </div>
    """