from __future__ import annotations

from typing import Any, Dict, List

import json
import math
from html import escape as _esc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from flask import url_for
from plotly.offline import plot as plotly_plot

from dashboard_services.ai.history_recap import get_league_season_summary
from dashboard_services.platform_api import get_bracket
from dashboard_services.service import playoff_bracket as _render_playoff_bracket
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
                    "winner_pts": _safe_float(winner["points"]),
                    "loser_pts": _safe_float(loser["points"]),
                    "margin": margin,
                }
            )

        if matchup_rows:
            closest = min(matchup_rows, key=lambda x: x["margin"])
            blowout = max(matchup_rows, key=lambda x: x["margin"])

            summary["closest_matchup"] = f"Week {closest['week']}: {closest['winner']} over {closest['loser']}"
            summary["closest_scores"] = f"{closest['winner_pts']:.1f}-{closest['loser_pts']:.1f}"
            summary["closest_margin"] = _safe_float(closest["margin"])

            summary["biggest_blowout"] = f"Week {blowout['week']}: {blowout['winner']} over {blowout['loser']}"
            summary["biggest_blowout_scores"] = f"{blowout['winner_pts']:.1f}-{blowout['loser_pts']:.1f}"
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
                "<i class='fa-solid fa-crown' aria-hidden='true'></i> Champion",
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


def _build_match_scores(winners_bracket, df_weekly, league) -> dict:
    """Build {match_id: {"t1_score": float, "t2_score": float}} from df_weekly.

    Each bracket round maps to a playoff week:
      round 1 → playoff_week_start, round 2 → playoff_week_start + 1, etc.
    We look up each team's points in that week to populate scores.
    """
    if df_weekly is None or getattr(df_weekly, "empty", True):
        return {}
    needed = {"roster_id", "week", "points"}
    if not needed.issubset(df_weekly.columns):
        return {}

    pw_start = _playoff_start_week(league)
    scores: dict = {}

    # Index: (roster_id_str, week_int) → points
    pts_lookup: dict = {}
    try:
        for _, row in df_weekly.iterrows():
            rid = row.get("roster_id")
            wk = row.get("week")
            pts = row.get("points")
            if rid is not None and wk is not None and pts is not None:
                pts_lookup[(str(rid), int(wk))] = float(pts)
    except Exception:
        return {}

    for m in winners_bracket:
        mid = m.get("m")
        rnd = m.get("r")
        t1 = m.get("t1")
        t2 = m.get("t2")
        if mid is None or rnd is None:
            continue
        week = pw_start + rnd - 1
        s1 = pts_lookup.get((str(t1), week)) if t1 is not None else None
        s2 = pts_lookup.get((str(t2), week)) if t2 is not None else None
        if s1 is not None or s2 is not None:
            scores[mid] = {"t1_score": s1, "t2_score": s2}

    return scores


def _get_history_bracket_html(history_ctx: dict) -> str:
    """Build the playoff bracket HTML for the history standings card."""
    platform = history_ctx.get("platform", "sleeper")
    season = history_ctx.get("season")
    league_id = history_ctx.get("resolved_league_id") or history_ctx.get("league_id") or ""
    roster_map = history_ctx.get("roster_map") or {}
    league = history_ctx.get("league") or {}
    df_weekly = history_ctx.get("df_weekly")

    try:
        wb = get_bracket(platform, league_id, "winners", season) or []
    except Exception:
        wb = []

    if not wb:
        return ""

    roster_avatar_map = {}
    roster_name_map = dict(roster_map)
    match_scores = _build_match_scores(wb, df_weekly, league)

    bracket_html = _render_playoff_bracket(
        wb,
        roster_name_map=roster_name_map,
        roster_avatar_map=roster_avatar_map,
        match_scores=match_scores,
    )

    if bracket_html.strip().startswith("<div class='po-empty'>"):
        return ""

    return bracket_html


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
                window.brEmptyState(result, { icon: 'search', title: 'No head-to-head yet', message: 'These two managers have never faced each other.', compact: true });
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
              window.brErrorState(result, 'Could not load rivalry data.', function() { goBtn.click(); }, { compact: true });
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


_WRAPPED_PLAYER_CACHE: dict = {}


def _wrapped_player_leaders(history_ctx: dict) -> dict:
    """Aggregate each player's regular-season fantasy production for the league
    from the weekly boxscores, and return {'mvp': entry, 'by_pos': {QB/RB/WR/TE:
    entry}} where entry is {name, pos, nfl, pts, ppg}. Cached per league+season
    (completed seasons are static). Returns {} if unavailable."""
    platform = str(history_ctx.get("platform") or "sleeper")
    league_id = str(history_ctx.get("resolved_league_id") or history_ctx.get("league_id") or "")
    season = history_ctx.get("season")
    players_map = history_ctx.get("players_map") or history_ctx.get("players_index") or {}
    if not league_id or not players_map:
        return {}
    ckey = (platform, league_id, str(season))
    if ckey in _WRAPPED_PLAYER_CACHE:
        return _WRAPPED_PLAYER_CACHE[ckey]

    try:
        from concurrent.futures import ThreadPoolExecutor
        from dashboard_services.platform_api import get_matchups

        # Sum the FULL season the league played, not just the fantasy regular
        # season. "Top producer of the year" should match the player card's
        # season total, which includes the fantasy-playoff weeks -- stopping at
        # playoff-start made a full-PPR total read like standard scoring (e.g. a
        # 24.5-PPG back showing ~344 over 14 weeks instead of ~416 over 17).
        # Bound by the last week the league actually has matchup data so we don't
        # fetch dead weeks.
        _df_w = history_ctx.get("df_weekly")
        try:
            _last_wk = int(_df_w["week"].max()) if _df_w is not None and not _df_w.empty else 0
        except Exception:
            _last_wk = 0
        if _last_wk < 2:
            _po = _playoff_start_week(history_ctx.get("league") or {}) or 15
            _last_wk = min(18, _po + 2)
        weeks = list(range(1, _last_wk + 1))

        def _fetch(w):
            try:
                return get_matchups(platform, league_id, w, season) or []
            except Exception:
                return []

        totals: dict[str, list] = {}  # pid -> [points, games_played]
        # Fetch every week's boxscore in a single concurrent batch (capped at 16)
        # so covering the full season instead of just the regular season doesn't
        # add a round-trip of latency to this once-per-league-season build.
        with ThreadPoolExecutor(max_workers=min(len(weeks), 16)) as pool:
            for mus in pool.map(_fetch, weeks):
                for m in mus:
                    for pid, pts in (m.get("players_points") or {}).items():
                        pid = str(pid)
                        if pid in ("", "0"):
                            continue
                        pv = float(pts or 0)
                        e = totals.setdefault(pid, [0.0, 0])
                        e[0] += pv
                        if pv != 0:
                            e[1] += 1

        _POS = {"QB", "RB", "WR", "TE"}
        mvp = None
        by_pos: dict = {}
        for pid, (pts, games) in totals.items():
            p = players_map.get(pid) or {}
            name = p.get("name") or p.get("full_name")
            if not name:
                continue
            # players_map stores position under "pos"; players_index uses "position".
            pos = str(p.get("pos") or p.get("position") or "").upper()
            nfl = str(p.get("team") or p.get("nfl") or "")
            entry = {"name": str(name), "pos": pos, "nfl": nfl,
                     "pts": round(pts, 1), "ppg": round(pts / games, 1) if games else 0.0}
            if mvp is None or pts > mvp["pts"]:
                mvp = entry
            if pos in _POS and (pos not in by_pos or pts > by_pos[pos]["pts"]):
                by_pos[pos] = entry

        leaders = {"mvp": mvp, "by_pos": by_pos}
        _WRAPPED_PLAYER_CACHE[ckey] = leaders
        return leaders
    except Exception:
        return {}


def _wrapped_activity(history_ctx: dict) -> dict:
    """Season transaction activity from ctx['activity_df']: total trades, the
    most active trader (owner involved in the most trades), and the waiver-wire
    leader (most adds). Returns {} when there's no activity data."""
    try:
        adf = history_ctx.get("activity_df")
        if adf is None or getattr(adf, "empty", True) or "kind" not in getattr(adf, "columns", []):
            return {}
        total_trades = 0
        trades_by_owner: dict = {}
        waivers_by_owner: dict = {}
        for _, row in adf.iterrows():
            kind = row.get("kind")
            data = row.get("data") or {}
            if kind == "trade":
                total_trades += 1
                for tm in (data.get("teams") or []):
                    nm = str(tm.get("name") or "").strip()
                    if nm:
                        trades_by_owner[nm] = trades_by_owner.get(nm, 0) + 1
            elif kind == "waiver":
                nm = str(data.get("name") or "").strip()
                if nm:
                    adds = data.get("adds") or []
                    waivers_by_owner[nm] = waivers_by_owner.get(nm, 0) + (len(adds) if isinstance(adds, list) else 1)
        out: dict = {"total_trades": total_trades}
        if trades_by_owner:
            o = max(trades_by_owner.items(), key=lambda x: x[1])
            out["top_trader"] = {"owner": o[0], "n": o[1]}
        if waivers_by_owner:
            w = max(waivers_by_owner.items(), key=lambda x: x[1])
            out["top_waiver"] = {"owner": w[0], "n": w[1]}
        return out
    except Exception:
        return {}


def _wrapped_luck(history_ctx: dict):
    """(lucky, unlucky) each as (owner, luck_delta) from the all-play luck index
    (actual wins minus expected wins), or (None, None). Same metric as the
    Standings 'Luck' column so the Wrapped agrees with the rest of the app."""
    try:
        from utils.all_play import all_play_analysis
        df = history_ctx.get("df_weekly")
        if df is None or df.empty or not {"owner", "points", "week"}.issubset(df.columns):
            return None, None
        df = _filtered_season_df(df)
        if df is None or df.empty:
            return None, None
        df = df.copy()
        if "finalized" in df.columns:
            df = df[df["finalized"] == True]
        df["week"] = pd.to_numeric(df["week"], errors="coerce")

        weekly_scores: dict = {}
        for wk, g in df.groupby("week"):
            weekly_scores[int(wk)] = {str(r["owner"]): float(r["points"] or 0) for _, r in g.iterrows()}

        actual: dict = {}
        has_pa = "points_against" in df.columns
        for _, r in df.iterrows():
            o = str(r["owner"])
            pf = float(r["points"] or 0)
            pa = float(r.get("points_against") or 0) if has_pa else 0.0
            actual[o] = actual.get(o, 0.0) + (1.0 if pf > pa else 0.5 if pf == pa else 0.0)

        ana = all_play_analysis(weekly_scores, actual)
        if not ana:
            return None, None
        items = [(t, d["luck_delta"]) for t, d in ana.items()]
        return max(items, key=lambda x: x[1]), min(items, key=lambda x: x[1])
    except Exception:
        return None, None


def _build_wrapped_slides(history_ctx: dict, summary: dict, league_name: str, season,
                          include_players: bool = True) -> list:
    """Ordered 'Season Wrapped' story slides built from the season summary and
    per-player leaders. Each slide: {kind, eyebrow, big, num, dp, suffix, label,
    sub} (plus optional 'rows' for a list slide).

    include_players=False skips the MVP / position-leader slides, which are the
    only ones that need per-week boxscore fetches. The page render uses that
    cheap path just to decide whether to show the launcher; the lazy /wrapped
    endpoint builds the full deck."""
    df_weekly = history_ctx.get("df_weekly", pd.DataFrame())
    league = history_ctx.get("league") or {}

    def _txt(kind, eyebrow, big, label, sub):
        return {"kind": kind, "eyebrow": eyebrow, "big": big, "num": False,
                "dp": 0, "suffix": "", "label": label, "sub": sub}

    def _num(kind, eyebrow, val, dp, suffix, label, sub):
        return {"kind": kind, "eyebrow": eyebrow, "big": f"{val:.{dp}f}", "num": True,
                "dp": dp, "suffix": suffix, "label": label, "sub": sub}

    slides = [{**_txt("intro", f"{season} SEASON", league_name, "Wrapped",
                      "A look back at the year that was"),
               "bgword": f"'{str(season)[-2:]}"}]

    if summary.get("top_scorer_value"):
        slides.append({**_num("topscore", "MOST POINTS ON THE YEAR",
                              summary["top_scorer_value"], 1, " PTS",
                              summary.get("top_scorer_team", "-"),
                              f"{summary.get('top_scorer_avg', 0):.1f} avg per week, the league's top scoring machine"),
                       "bgword": str(int(summary["top_scorer_value"]))})

    # ── Player awards: season MVP + the best at each position ──────────────────
    # These are the only slides that need per-week boxscore fetches, so the
    # cheap availability check skips them.
    if include_players:
        leaders = _wrapped_player_leaders(history_ctx)
        mvp = (leaders or {}).get("mvp")
        if mvp and mvp.get("pts"):
            _meta = " · ".join(x for x in [mvp.get("pos"), mvp.get("nfl")] if x)
            slides.append({**_num("mvp", "LEAGUE MVP", float(mvp["pts"]), 1, " PTS",
                                  mvp["name"],
                                  f"{_meta} · {mvp.get('ppg', 0):.1f} per game, the season's top fantasy producer"),
                           "bgword": str(int(float(mvp["pts"])))})

        by_pos = (leaders or {}).get("by_pos") or {}
        pos_rows = [(p, by_pos[p]["name"], f"{by_pos[p]['pts']:.1f}")
                    for p in ("QB", "RB", "WR", "TE") if by_pos.get(p)]
        if len(pos_rows) >= 3:
            slides.append({"kind": "posleaders", "eyebrow": "TOP AT EACH POSITION",
                           "num": False, "big": "", "dp": 0, "suffix": "", "label": "",
                           "sub": "The points leader at every spot", "rows": pos_rows,
                           "bgword": "TOP"})

    # Season records: biggest single week + hottest streak, grouped.
    _rec_rows = []
    if summary.get("highest_week_value"):
        _rec_rows.append(("BIG WEEK", summary.get("highest_week_team", "-"),
                          f"{summary['highest_week_value']:.1f}"))
    _streak_owner, _streak_len = _wrapped_longest_win_streak(df_weekly, league)
    if _streak_len >= 3:
        _rec_rows.append(("HOT STREAK", _streak_owner, f"{_streak_len} W"))
    if _rec_rows:
        slides.append({"kind": "records", "eyebrow": "SEASON RECORDS", "num": False,
                       "big": "", "dp": 0, "suffix": "", "label": "",
                       "sub": "The high-water marks of the year", "rows": _rec_rows,
                       "bgword": "BEST"})

    if summary.get("biggest_blowout_margin"):
        _b = _num("blowout", "BIGGEST BLOWOUT",
                  summary["biggest_blowout_margin"], 1, " PTS",
                  summary.get("biggest_blowout", "-"),
                  "The most lopsided result of the year")
        _b["bgword"] = str(int(summary["biggest_blowout_margin"]))
        if summary.get("biggest_blowout_scores"):
            _b["scoreline"] = summary["biggest_blowout_scores"]
        slides.append(_b)

    if summary.get("closest_margin"):
        _c = _num("nailbiter", "CLOSEST GAME",
                  summary["closest_margin"], 1, " PTS",
                  summary.get("closest_matchup", "-"),
                  "Decided by the slimmest margin all season")
        _c["bgword"] = f"{summary['closest_margin']:.1f}"
        if summary.get("closest_scores"):
            _c["scoreline"] = summary["closest_scores"]
        slides.append(_c)

    # Luck index (all-play luck_delta): luckiest + unluckiest, grouped.
    _lucky, _unlucky = _wrapped_luck(history_ctx)
    _luck_rows = []
    if _lucky and _lucky[1] >= 1.0:
        _luck_rows.append(("LUCKIEST", _lucky[0], f"{_lucky[1]:+.1f} W"))
    if _unlucky and _unlucky[1] <= -1.0:
        _luck_rows.append(("UNLUCKIEST", _unlucky[0], f"{_unlucky[1]:.1f} W"))
    if _luck_rows:
        slides.append({"kind": "luck", "eyebrow": "THE LUCK INDEX", "num": False,
                       "big": "", "dp": 0, "suffix": "", "label": "",
                       "sub": "Wins above or below what the scoring earned", "rows": _luck_rows,
                       "bgword": "LUCK"})

    # League activity: total trades, most active trader, waiver-wire leader.
    act = _wrapped_activity(history_ctx)
    _act_rows = []
    if act.get("total_trades"):
        _act_rows.append(("TRADES", f"{act['total_trades']} deals", ""))
    _tt = act.get("top_trader")
    if _tt and _tt.get("n"):
        _act_rows.append(("TOP TRADER", _tt["owner"], f"{_tt['n']}"))
    _tw = act.get("top_waiver")
    if _tw and _tw.get("n"):
        _act_rows.append(("WAIVERS", _tw["owner"], f"{_tw['n']}"))
    if _act_rows:
        slides.append({"kind": "activity", "eyebrow": "LEAGUE ACTIVITY", "num": False,
                       "big": "", "dp": 0, "suffix": "", "label": "",
                       "sub": "Who worked the phones and the wire", "rows": _act_rows,
                       "bgword": "MOVES"})

    if summary.get("runner_up") and summary.get("runner_up") not in ("-", None):
        slides.append({**_txt("runnerup", "RUNNER-UP", summary["runner_up"], "So close",
                              f"Finished {summary.get('runner_up_record', '')}, one game short"),
                       "bgword": "2ND"})

    if summary.get("champion") and summary.get("champion") not in ("-", None):
        slides.append({**_txt("champion", f"{season} CHAMPION", summary["champion"],
                              "", "Took the crown when it mattered most"),
                       "bgword": "CHAMP",
                       "record": str(summary.get("champion_record", "") or "")})

    return slides


def _wrapped_overlay_markup(slides: list, share_data: dict | None = None,
                            season=None, ns: str = "wrapped",
                            footer_label: str | None = None) -> str:
    """Return the Season Wrapped overlay markup (no launch button, no script) so
    it can be fetched and injected lazily. '' when there isn't enough to tell a
    story. Editorial layout: left-aligned content on a strict margin, a kicker
    rule above each headline, a giant outlined background word, and a broadcast
    footer bug on every slide.

    ns namespaces every element id (``wrapped`` -> ``{ns}``) so a Weekly
    Wrapped overlay can share a page with the Season Wrapped one; the bootstrap
    JS is namespaced the same way. footer_label overrides the footer bug text
    (Weekly Wrapped passes ``WEEK N``); intro slides may carry ``intro_word``
    HTML to replace the default ``SEASON<br>WRAPPED`` wordmark."""
    if len(slides) < 3:
        return ""

    bars = "".join("<span class='wrapped-bar'><i></i></span>" for _ in slides)
    season_txt = _esc(str(season)) if season not in (None, "") else ""
    if footer_label is not None:
        foot_txt = footer_label
    elif season_txt:
        foot_txt = f"{season_txt} SEASON"
    else:
        foot_txt = ""
    foot = (
        "<div class='wrapped-foot'>"
        "<img src='/static/BR_Logo_dark.png?v=6c0c4828' alt=''>"
        "<span class='wrapped-foot-line'></span>"
        f"<span class='wrapped-foot-season'>{_esc(foot_txt)}</span>"
        "</div>"
    )

    slide_html = []
    for s in slides:
        kind = s["kind"]
        kicker = (f"<div class='wrapped-kicker'><span class='wrapped-kicker-rule'></span>"
                  f"<span class='wrapped-kicker-txt'>{_esc(str(s['eyebrow']))}</span></div>")
        bgword = (f"<div class='wrapped-bgword' aria-hidden='true'>{_esc(str(s['bgword']))}</div>"
                  if s.get("bgword") else "")
        sub = (f"<div class='wrapped-divider'></div>"
               f"<div class='wrapped-sub'>{_esc(str(s['sub']))}</div>")

        if kind == "intro":
            _intro_word = s.get("intro_word") or "SEASON<br>WRAPPED"
            body = (
                "<img src='/static/BR_Logo_dark.png?v=6c0c4828' alt='BR Fantasy' class='wrapped-intro-logo'>"
                f"<div class='wrapped-league'>{_esc(str(s['big']))}</div>"
                f"<div class='wrapped-word'>{_intro_word}</div>"
                + sub
            )
        elif kind == "champion":
            record = str(s.get("record") or "")
            record_html = (
                "<div class='wrapped-record'>"
                + (f"<span class='wrapped-record-badge'>{_esc(record)}</span>" if record else "")
                + "<span class='wrapped-record-txt'>League champion</span></div>"
            )
            body = (
                "<div class='wrapped-rays' aria-hidden='true'></div>"
                "<i class='fa-solid fa-crown wrapped-crown' aria-hidden='true'></i>"
                + kicker
                + f"<div class='wrapped-cname'>{_esc(str(s['big']))}</div>"
                + record_html + sub
            )
        elif kind == "recap":
            # The shareable summary card, as a slide: champion block on top, then
            # the same highlight rows the Share card draws.
            rd = s.get("recap") or {}
            champ_html = ""
            if rd.get("champion"):
                rec = str(rd.get("champion_record") or "")
                champ_html = (
                    "<div class='wrapped-recap-champ'>"
                    "<i class='fa-solid fa-crown wrapped-recap-crown' aria-hidden='true'></i>"
                    f"<span class='wrapped-recap-champ-name'>{_esc(str(rd['champion']))}</span>"
                    + (f"<span class='wrapped-record-badge'>{_esc(rec)}</span>" if rec else "")
                    + "<span class='wrapped-recap-champ-lbl'>Champion</span></div>"
                )
            hl = "".join(
                "<div class='wrapped-row'>"
                f"<div class='wrapped-row-k'>{_esc(str(h.get('k', '')))}</div>"
                "<div class='wrapped-row-m'>"
                f"<span class='wrapped-row-n'>{_esc(str(h.get('n', '')))}</span>"
                + (f"<span class='wrapped-row-v'>{_esc(str(h.get('v', '')))}</span>" if h.get('v') not in (None, "") else "")
                + "</div></div>"
                # Cap at 4 on the slide (the Share card still draws up to 5) so a
                # champion block + rows never overflow a small phone's story slide.
                for h in (rd.get("highlights") or [])[:4]
            )
            body = kicker + champ_html + f"<div class='wrapped-rows'>{hl}</div>" + sub
        elif s.get("rows"):
            rows = "".join(
                "<div class='wrapped-row'>"
                f"<div class='wrapped-row-k'>{_esc(str(k))}</div>"
                "<div class='wrapped-row-m'>"
                f"<span class='wrapped-row-n'>{_esc(str(n))}</span>"
                + (f"<span class='wrapped-row-v'>{_esc(str(v))}</span>" if v not in (None, "") else "")
                + "</div></div>"
                for k, n, v in s["rows"]
            )
            body = kicker + f"<div class='wrapped-rows'>{rows}</div>" + sub
        elif s.get("preview_teams"):
            # Next week's Game of the Week (a preview: no winner yet). Neutral
            # stacked hero, the projected win-% split bar, projected scoreline.
            _teams = [str(t) for t in (s.get("preview_teams") or [])[:2]]
            _hero = "<div class='wrapped-vs-hero'>"
            for _i, _t in enumerate(_teams):
                if _i:
                    _hero += "<div class='wrapped-vs-hero-x'>vs</div>"
                _hero += f"<div class='wrapped-duel-t'>{_esc(_t)}</div>"
            _hero += "</div>"
            _winbar = ""
            _wpa = s.get("win_prob_a")
            if isinstance(_wpa, (int, float)):
                _wpi = int(round(_wpa))
                _rpi = 100 - _wpi
                _winbar = (
                    "<div class='wrapped-winbar-row'>"
                    f"<span class='wrapped-winbar-pct'>{_wpi}%</span>"
                    "<div class='wrapped-winbar'>"
                    f"<i style='width:{_wpi}%'></i>"
                    "</div>"
                    f"<span class='wrapped-winbar-pct wrapped-winbar-pct-r'>{_rpi}%</span>"
                    "</div>")
            _why = str(s.get("why") or "").strip()
            _why_html = (f"<div class='wrapped-why'>{_esc(_why)}</div>"
                         if _why else "")
            body = (
                kicker
                + _why_html
                + _hero
                + _winbar
                + sub
            )
        elif s["num"]:
            unit = str(s.get("suffix") or "").strip()
            score_html = ""
            _sl = str(s.get("scoreline") or "")
            if "-" in _sl:
                _w, _l = _sl.split("-", 1)
                score_html = ("<div class='wrapped-scores'>"
                              f"<span class='wrapped-score-w'>{_esc(_w.strip())}</span>"
                              "<span class='wrapped-score-x'>VS</span>"
                              f"<span class='wrapped-score-l'>{_esc(_l.strip())}</span></div>")
            body = (
                kicker
                + "<div class='wrapped-num'>"
                  f"<span class='wrapped-big' data-w-count='{s['big']}' data-w-dp='{s['dp']}'>0</span>"
                + (f"<span class='wrapped-unit'>{_esc(unit)}</span>" if unit else "")
                + "</div>"
                + (f"<div class='wrapped-name'>{_esc(str(s['label']))}</div>" if s["label"] else "")
                + score_html + sub
            )
        else:
            # Text slide (runner-up): big name with its short label under it.
            body = (
                kicker
                + f"<div class='wrapped-cname'>{_esc(str(s['big']))}</div>"
                + (f"<div class='wrapped-name'>{_esc(str(s['label']))}</div>" if s["label"] else "")
                + sub
            )

        slide_html.append(
            f"<section class='wrapped-slide' data-kind='{kind}'>{bgword}{body}{foot}</section>"
        )

    share_json = json.dumps(share_data or {}).replace("</", "<\\/")
    return f"""
    <div class="wrapped-overlay" id="{ns}Overlay" hidden aria-hidden="true">
      <div class="wrapped-progress">{bars}</div>
      <button type="button" class="wrapped-share" id="{ns}Share" aria-label="Share">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg><span>Share</span>
      </button>
      <button type="button" class="wrapped-link" id="{ns}Link" aria-label="Copy shareable link">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg><span>Link</span>
      </button>
      <button type="button" class="wrapped-close" id="{ns}Close" aria-label="Close">&times;</button>
      <div class="wrapped-stage" id="{ns}Stage">{''.join(slide_html)}</div>
      <button type="button" class="wrapped-tap wrapped-tap-prev" id="{ns}Prev" aria-label="Previous"></button>
      <button type="button" class="wrapped-tap wrapped-tap-next" id="{ns}Next" aria-label="Next"></button>
      <div class="wrapped-hint">Tap to advance · Esc to close</div>
      <script type="application/json" id="{ns}ShareData">{share_json}</script>
    </div>
    """


def _wrapped_share_data(slides: list, summary: dict, league_name: str, season) -> dict:
    """Curated highlights for the single shareable 'Season Wrapped' summary card,
    drawn client-side to a canvas. Built from the same slides so the card agrees
    with the deck."""
    by_kind = {s.get("kind"): s for s in slides}

    def _hi(k, n, v):
        return {"k": str(k), "n": str(n), "v": str(v)}

    highlights: list = []
    if summary.get("top_scorer_value"):
        highlights.append(_hi("TOP SCORER", summary.get("top_scorer_team", "-"),
                              f"{summary['top_scorer_value']:.1f}"))
    mvp = by_kind.get("mvp")
    if mvp and mvp.get("label"):
        highlights.append(_hi("LEAGUE MVP", mvp["label"], f"{mvp['big']}"))
    if summary.get("biggest_blowout_margin"):
        highlights.append(_hi("BIGGEST BLOWOUT", summary.get("biggest_blowout", "-"),
                              f"{summary['biggest_blowout_margin']:.1f}"))
    if summary.get("closest_margin"):
        highlights.append(_hi("CLOSEST GAME", summary.get("closest_matchup", "-"),
                              f"{summary['closest_margin']:.1f}"))
    luck = by_kind.get("luck")
    if luck and luck.get("rows"):
        _k, _n, _v = luck["rows"][0]
        highlights.append(_hi("LUCKIEST", _n, _v))

    champion = summary.get("champion")
    if champion in ("-", None):
        champion = None
    return {
        "league": str(league_name),
        "season": str(season),
        "champion": champion,
        "champion_record": str(summary.get("champion_record", "") or ""),
        "highlights": highlights[:5],
    }


def render_history_wrapped_overlay(history_ctx: dict, selected_history_season) -> str:
    """Build the full Season Wrapped overlay markup (including the boxscore-backed
    MVP / position slides). Called by the lazy /wrapped endpoint on first open."""
    league = history_ctx.get("league") or {}
    league_name = league.get("name") or "League History"
    summary = history_ctx.get("summary") or _build_summary(history_ctx)
    slides = _build_wrapped_slides(history_ctx, summary, league_name, selected_history_season)
    share_data = _wrapped_share_data(slides, summary, league_name, selected_history_season)
    # Finale: the shareable recap card as a slide in the deck itself, so the deck
    # ends on the same one-card summary you get from the Share button.
    if share_data.get("highlights") or share_data.get("champion"):
        _season_txt = str(selected_history_season) if selected_history_season not in (None, "") else ""
        slides.append({
            "kind": "recap", "num": False, "big": "", "dp": 0, "suffix": "", "label": "",
            "eyebrow": f"{_season_txt} RECAP".strip() or "SEASON RECAP",
            "sub": "Tap Share to send it to the group chat",
            "bgword": "RECAP", "recap": share_data,
        })
    return _wrapped_overlay_markup(slides, share_data, season=selected_history_season)


# ── Weekly Wrapped ────────────────────────────────────────────────────────────
# One-week story decks for the weekly hub: intro, highest team score, biggest
# blowout, closest game, the week's top fantasy player + position leaders + dud
# (boxscore-backed, lazy like the season MVP slide), and a shareable recap
# finale. The overlay renderer, launcher, and bootstrap are shared with Season
# Wrapped via the ns="weekly-wrapped" namespace.


def _mfield(m, key):
    """Read a matchup field whether the provider returned a dict or a
    MatchupRow dataclass."""
    try:
        return m.get(key)
    except AttributeError:
        return getattr(m, key, None)


def _wrapped_weekly_player_leaders(ctx: dict, week) -> dict:
    """Per-player fantasy production for a single week from that week's
    boxscores. Returns {'top': entry, 'by_pos': {QB/RB/WR/TE/K/DEF: entry},
    'dud': entry} where entry is {name, pos, nfl, pts}. 'dud' is the
    lowest-scoring starter (needs the provider's starters list; skipped when
    unavailable). One week, one fetch, per request -- no cache. {} when
    unavailable."""
    platform = str(ctx.get("platform") or "sleeper")
    league_id = str(ctx.get("resolved_league_id") or ctx.get("league_id") or "")
    season = ctx.get("season")
    players_map = ctx.get("players_map") or {}
    if not league_id or not players_map:
        return {}
    try:
        from dashboard_services.platform_api import get_matchups
        mus = get_matchups(platform, league_id, int(week), season) or []
    except Exception:
        return {}

    _POS = ("QB", "RB", "WR", "TE", "K", "DEF")
    top = None
    by_pos: dict = {}
    dud = None
    for m in mus:
        pp = _mfield(m, "players_points") or {}
        starters = {str(s) for s in (_mfield(m, "starters") or [])}
        for pid, pts in pp.items():
            pid = str(pid)
            if pid in ("", "0"):
                continue
            try:
                pv = float(pts or 0)
            except (TypeError, ValueError):
                continue
            p = players_map.get(pid) or {}
            name = p.get("name") or p.get("full_name")
            if not name:
                continue
            pos = str(p.get("pos") or p.get("position") or "").upper()
            nfl = str(p.get("team") or p.get("nfl") or "")
            entry = {"name": str(name), "pos": pos, "nfl": nfl, "pts": round(pv, 1)}
            if top is None or pv > top["pts"]:
                top = entry
            if pos in _POS and (pos not in by_pos or pv > by_pos[pos]["pts"]):
                by_pos[pos] = entry
            if pid in starters and pv > 0 and (dud is None or pv < dud["pts"]):
                dud = entry
    return {"top": top, "by_pos": by_pos, "dud": dud}


def _next_week_gotw_game(ctx: dict, week: int, platform: str, league_id: str,
                         season):
    """The ``game_of_the_week`` dict for week+1 (next week's featured matchup).

    Mirrors the weekly hub's deterministic pick: honors the cached GOTW
    selection when present, otherwise computes it through the recap payload
    builder and caches the selection so the hub badge agrees. Returns None
    when next week's matchups/projections can't be built. Lazy-path only --
    it fetches next-week matchup previews and projections.
    """
    try:
        from dashboard_services.ai.weekly_recap import (
            build_weekly_recap_payload,
            get_cached_gotw_selection,
            _gotw_cache_key,
            save_cached_ai_text,
        )
        from app import _build_next_week_ctx
    except Exception:
        return None
    try:
        next_week = int(week) + 1
        df_weekly = ctx.get("df_weekly")
        matchups_by_week = ctx.get("matchups_by_week") or {}
        roster_map = ctx.get("roster_map") or {}
        if df_weekly is None or getattr(df_weekly, "empty", True):
            return None
        team_by_rid = {str(rid): name for rid, name in roster_map.items()}
        if not team_by_rid:
            return None
        league = dict(ctx.get("league") or {})
        if ctx.get("league_settings"):
            merged = dict(ctx["league_settings"])
            merged.update(league.get("settings") or {})
            league["settings"] = merged
        playoff_start = int((league.get("settings") or {}).get("playoff_week_start") or 14)

        sel = get_cached_gotw_selection(str(platform or "sleeper"),
                                        str(league_id or ""), season, next_week)
        want = {str(x) for x in ((sel or {}).get("roster_ids") or []) if str(x)}

        nctx = _build_next_week_ctx(ctx, next_week, playoff_start,
                                    str(league_id or ""), season,
                                    str(platform or "sleeper"), team_by_rid)
        if not nctx:
            return None
        payload = build_weekly_recap_payload(
            df_weekly, matchups_by_week, int(week), team_by_rid,
            league, next_week_ctx=nctx)
        game = (payload.get("next_week_preview") or {}).get("game_of_the_week") or {}
        if not game or not game.get("team_a") or not game.get("team_b"):
            return None
        rids = [str(game.get("roster_id_a") or ""), str(game.get("roster_id_b") or "")]
        if want and want != set(rids):
            return None
        if not want and all(rids):
            # No cached pick yet: store this deterministic one so the hub
            # badge and the wrapped teaser agree.
            try:
                save_cached_ai_text(
                    _gotw_cache_key(platform, league_id, season, next_week), "",
                    metadata={"gotw_selection": {
                        "platform": str(platform or "").lower(),
                        "league_id": str(league_id or ""),
                        "season": str(season),
                        "source_week": int(week),
                        "target_week": int(next_week),
                        "matchup_id": game.get("matchup_id"),
                        "roster_ids": rids}})
            except Exception:
                pass
        game["target_week"] = next_week
        return game
    except Exception:
        return None


def _build_weekly_wrapped_slides(ctx: dict, league_name: str, season, week,
                                 include_players: bool = True) -> list:
    """Ordered 'Weekly Wrapped' story slides for one completed week. Each slide:
    {kind, eyebrow, big, num, dp, suffix, label, sub} (plus optional 'rows').

    Slides with no data are skipped; the overlay needs >= 3 to render.
    include_players=False skips the boxscore-backed slides (top player /
    position leaders / dud), so the hub can cheaply decide whether to show the
    launcher; the lazy /wrapped endpoint builds the full deck."""
    try:
        week = int(week)
    except (TypeError, ValueError):
        return []

    wdf = ctx.get("df_weekly")
    if wdf is None or getattr(wdf, "empty", True) or "week" not in wdf.columns:
        return []
    wdf = wdf.copy()
    wdf = wdf[pd.to_numeric(wdf["week"], errors="coerce") == week]
    if wdf.empty:
        return []
    # Only completed games count -- never wrap projections.
    if "finalized" in wdf.columns:
        fin = wdf[wdf["finalized"] == True]
        if not fin.empty:
            wdf = fin
    if "points" not in wdf.columns:
        return []
    wdf["points"] = pd.to_numeric(wdf["points"], errors="coerce").fillna(0.0)
    if not (wdf["points"] > 0).any():
        return []

    def _txt(kind, eyebrow, big, label, sub):
        return {"kind": kind, "eyebrow": eyebrow, "big": big, "num": False,
                "dp": 0, "suffix": "", "label": label, "sub": sub}

    def _num(kind, eyebrow, val, dp, suffix, label, sub):
        return {"kind": kind, "eyebrow": eyebrow, "big": f"{val:.{dp}f}", "num": True,
                "dp": dp, "suffix": suffix, "label": label, "sub": sub}

    slides = [{**_txt("intro", f"WEEK {week}", str(league_name), "Wrapped",
                      "The week that was, in stories"),
               "bgword": f"W{week}", "intro_word": "WEEKLY<br>WRAPPED"}]

    hi = wdf.loc[wdf["points"].idxmax()]
    hi_pts = float(hi.get("points") or 0)
    if hi_pts > 0:
        slides.append({**_num("topscore", f"HIGHEST SCORE · WEEK {week}",
                              hi_pts, 1, " PTS", str(hi.get("owner", "-")),
                              "The week's biggest team total"),
                       "bgword": str(int(hi_pts))})

    lo = wdf.loc[wdf["points"].idxmin()]
    lo_pts = float(lo.get("points") or 0)
    if 0 < lo_pts < hi_pts:
        slides.append({**_num("lowscore", f"LOWEST SCORE · WEEK {week}",
                              lo_pts, 1, " PTS", str(lo.get("owner", "-")),
                              "The week's quietest team total"),
                       "bgword": str(int(lo_pts))})

    # Pair the week's head-to-head results via matchup_id.
    matchup_rows = []
    if "matchup_id" in wdf.columns:
        # Scalar (not list) grouper: yields plain scalar keys on every pandas
        # version. A one-element list grouper unpacks differently across
        # versions (tuple key on new pandas, scalar key on old).
        for _, grp in wdf.groupby("matchup_id"):
            if len(grp) != 2:
                continue
            ordered = grp.sort_values("points", ascending=False).reset_index(drop=True)
            winner, loser = ordered.iloc[0], ordered.iloc[1]
            wp, lp = float(winner["points"] or 0), float(loser["points"] or 0)
            if wp <= 0 and lp <= 0:
                continue
            matchup_rows.append({
                "winner": str(winner.get("owner", "-")),
                "loser": str(loser.get("owner", "-")),
                "winner_pts": wp, "loser_pts": lp, "margin": abs(wp - lp),
                "roster_ids": ({str(r) for r in grp["roster_id"]}
                               if "roster_id" in grp.columns else set()),
            })

    if matchup_rows:
        bo = max(matchup_rows, key=lambda x: x["margin"])
        _b = _num("blowout", "BIGGEST BLOWOUT", bo["margin"], 1, " PTS",
                  f"{bo['winner']} over {bo['loser']}",
                  "The week's most lopsided result")
        _b["bgword"] = str(int(bo["margin"]))
        _b["scoreline"] = f"{bo['winner_pts']:.1f}-{bo['loser_pts']:.1f}"
        slides.append(_b)

    if len(matchup_rows) >= 2:
        nb = min(matchup_rows, key=lambda x: x["margin"])
        _c = _num("nailbiter", "CLOSEST GAME", nb["margin"], 1, " PTS",
                  f"{nb['winner']} over {nb['loser']}",
                  "Decided by the slimmest margin of the week")
        _c["bgword"] = f"{nb['margin']:.1f}"
        _c["scoreline"] = f"{nb['winner_pts']:.1f}-{nb['loser_pts']:.1f}"
        slides.append(_c)

    # ── Player awards: top scorer, best at each position, dud of the week ─────
    # The only slides that need a boxscore fetch, so the cheap availability
    # check skips them (and the next-week GOTW preview below, which fetches
    # next week's matchup previews + projections).
    if include_players:
        leaders = _wrapped_weekly_player_leaders(ctx, week)
        top = (leaders or {}).get("top")
        if top and top.get("pts"):
            _meta = " · ".join(x for x in [top.get("pos"), top.get("nfl")] if x)
            slides.append({**_num("topplayer", "TOP PLAYER OF THE WEEK",
                                  float(top["pts"]), 1, " PTS", top["name"],
                                  f"{_meta}, the week's top fantasy producer" if _meta
                                  else "The week's top fantasy producer"),
                           "bgword": str(int(float(top["pts"])))})

        by_pos = (leaders or {}).get("by_pos") or {}
        pos_rows = [(p, by_pos[p]["name"], f"{by_pos[p]['pts']:.1f}")
                    for p in ("QB", "RB", "WR", "TE", "K", "DEF") if by_pos.get(p)]
        if len(pos_rows) >= 3:
            slides.append({"kind": "posleaders", "eyebrow": "TOP AT EACH POSITION",
                           "num": False, "big": "", "dp": 0, "suffix": "", "label": "",
                           "sub": "The week's points leader at every spot",
                           "rows": pos_rows, "bgword": "TOP"})

        dud = (leaders or {}).get("dud")
        if dud and dud.get("pts"):
            _dmeta = " · ".join(x for x in [dud.get("pos"), dud.get("nfl")] if x)
            slides.append({**_num("dud", "DUD OF THE WEEK", float(dud["pts"]), 1,
                                  " PTS", dud["name"],
                                  f"{_dmeta}, the week's coldest starter" if _dmeta
                                  else "The week's coldest starter"),
                           "bgword": f"{float(dud['pts']):.1f}"})

    # ── Game of the week: a look AHEAD at next week's featured matchup ─────
    # The GOTW is a preview concept -- there is no winner yet. Closing teaser
    # of the deck, built from the same deterministic pick the hub badge and
    # the recap use. Lazy-build only (fetches next-week previews +
    # projections), so the hub's cheap launcher check skips it.
    if include_players:
        _game = _next_week_gotw_game(
            ctx, week, str(ctx.get("platform") or "sleeper"),
            str(ctx.get("resolved_league_id") or ctx.get("league_id") or ""),
            season)
        if _game and _game.get("team_a") and _game.get("team_b"):
            _nw = int(_game.get("target_week") or (week + 1))
            _ra, _rb = _game.get("rank_a"), _game.get("rank_b")
            _why = ""
            if _ra and _rb:
                _why = (f"#{_ra} ({_game.get('record_a') or '?'}) vs "
                        f"#{_rb} ({_game.get('record_b') or '?'})")
            _pa, _pb = _game.get("proj_a"), _game.get("proj_b")
            if isinstance(_pa, (int, float)) and isinstance(_pb, (int, float)):
                _sub = f"Projected {_pa:.1f}–{_pb:.1f}"
                _scoreline = f"{_pa:.1f}-{_pb:.1f}"
            else:
                _sub = "Next week's featured showdown"
                _scoreline = ""
            _g = {**_txt("gotw", f"WEEK {_nw} · GAME OF THE WEEK",
                         f"{_game['team_a']} vs {_game['team_b']}", "", _sub),
                  "bgword": "GOTW",
                  "preview_teams": [_game["team_a"], _game["team_b"]],
                  "win_prob_a": _game.get("win_prob_a"),
                  "why": _why}
            _g["scoreline"] = _scoreline
            slides.append(_g)

    return slides


def _wrapped_weekly_share_data(slides: list, league_name: str, season, week) -> dict:
    """Curated highlights for the shareable 'Weekly Wrapped' summary card,
    drawn client-side to a canvas. Built from the same slides so the card agrees
    with the deck. The painter keys off ``week`` for WEEK N branding."""
    by_kind = {s.get("kind"): s for s in slides}

    def _hi(k, n, v):
        return {"k": str(k), "n": str(n), "v": str(v)}

    highlights: list = []
    tp = by_kind.get("topplayer")
    if tp and tp.get("label"):
        highlights.append(_hi("TOP PLAYER", tp["label"], tp["big"]))
    gw = by_kind.get("gotw")
    if gw and gw.get("big"):
        highlights.append(_hi("GAME OF THE WEEK", gw["big"],
                              gw.get("scoreline") or ""))
    ts = by_kind.get("topscore")
    if ts and ts.get("label"):
        highlights.append(_hi("HIGH SCORE", ts["label"], ts["big"]))
    ls = by_kind.get("lowscore")
    if ls and ls.get("label"):
        highlights.append(_hi("LOW SCORE", ls["label"], ls["big"]))
    bo = by_kind.get("blowout")
    if bo and bo.get("label"):
        highlights.append(_hi("BIGGEST BLOWOUT", bo["label"], bo["big"]))
    nb = by_kind.get("nailbiter")
    if nb and nb.get("label"):
        highlights.append(_hi("CLOSEST GAME", nb["label"], nb["big"]))

    return {
        "league": str(league_name),
        "season": str(season),
        "week": int(week),
        "highlights": highlights[:5],
    }


def render_weekly_wrapped_overlay(ctx: dict, week) -> str:
    """Build the full Weekly Wrapped overlay markup (including the boxscore-backed
    top-player / position-leader / dud slides). Called by the lazy /wrapped
    endpoint on first open."""
    league = ctx.get("league") or {}
    league_name = league.get("name") or "League"
    season = ctx.get("season")
    week = int(week)
    slides = _build_weekly_wrapped_slides(ctx, league_name, season, week)
    share_data = _wrapped_weekly_share_data(slides, league_name, season, week)
    # Finale: the shareable recap card as a slide in the deck itself, so the deck
    # ends on the same one-card summary you get from the Share button.
    if share_data.get("highlights"):
        slides.append({
            "kind": "recap", "num": False, "big": "", "dp": 0, "suffix": "", "label": "",
            "eyebrow": f"WEEK {week} RECAP",
            "sub": "Tap Share to send it to the group chat",
            "bgword": "RECAP", "recap": share_data,
        })
    return _wrapped_overlay_markup(slides, share_data, ns="weekly-wrapped",
                                   footer_label=f"WEEK {week}")


def weekly_wrapped_url(ctx: dict, platform: str, season, league_id: str, week) -> str:
    """Cheap (no boxscore) availability check for the Weekly Wrapped launcher:
    the lazy endpoint URL when the week has enough finalized data for a >= 3
    slide deck, else ''. Used by the hub render and the week-change API so the
    button only appears for weeks with completed games."""
    try:
        league = ctx.get("league") or {}
        league_name = league.get("name") or "League"
        slides = _build_weekly_wrapped_slides(ctx, league_name, season, int(week),
                                              include_players=False)
        if len(slides) >= 3:
            return f"/api/weekly/{platform}/{season}/{league_id}/{int(week)}/wrapped"
    except Exception:
        pass
    return ""


def weekly_wrapped_launcher_html(ctx: dict, platform: str, season, league_id: str,
                                 week, league_name: str | None = None) -> str:
    """The Weekly Wrapped launch button + mount + namespaced bootstrap for the
    weekly hub. Hidden (display:none) when the week has no completed games; the
    hub's week-change JS re-points and re-shows it for scored weeks."""
    url = weekly_wrapped_url(ctx, platform, season, league_id, week)
    return _wrapped_launcher_html(url, ns="weekly-wrapped", label="Weekly Wrapped",
                                  hidden=not url)


def _wrapped_launcher_html(wrapped_url: str, ns: str = "wrapped",
                           label: str = "Season Wrapped",
                           hidden: bool = False) -> str:
    """The launch button + empty mount + bootstrap. The overlay itself is fetched
    from wrapped_url the first time the button is clicked, so the heavy boxscore
    aggregation never blocks the page render.

    ns namespaces the button/mount/overlay ids (and the bootstrap's lookups) so
    a Weekly Wrapped launcher can share a page with the Season Wrapped one.
    hidden renders the button with display:none for weeks with no completed
    games yet -- the week-change JS re-shows it when a scored week is picked."""
    _style = " style='display:none'" if hidden else ""
    return (
        f"<button type='button' class='wrapped-launch' id='{ns}Launch'{_style} "
        f"data-wrapped-url=\"{_esc(str(wrapped_url), quote=True)}\">"
        f"<i class='fa-solid fa-star'></i> {_esc(label)}</button>"
        f"<div id='{ns}Mount'></div>"
        f"<script>{_wrapped_bootstrap_js(ns)}</script>"
    )


def _wrapped_bootstrap_js(ns: str = "wrapped") -> str:
    """Bootstrap JS with every element id namespaced to ``ns`` (``wrapped`` by
    default), matching the ids ``_wrapped_overlay_markup(..., ns=ns)`` emits."""
    js = _WRAPPED_BOOTSTRAP_JS
    for _id in ("Launch", "Mount", "Overlay", "Stage", "Share", "Close",
                "Next", "Prev", "ShareData", "Link", "Toast"):
        js = js.replace(f"'wrapped{_id}'", f"'{ns}{_id}'")
    return js


def _wrapped_public_bootstrap_js(ns: str = "wrapped") -> str:
    """Auto-opening variant of the wrapped bootstrap for the public share page.

    The stored overlay HTML is already in the DOM (no launch button, no lazy
    fetch), so the launch-gating preamble and click handler are replaced with
    a direct openWrapped() call. Everything else -- story navigation, share
    card painting, copy-link (which just copies location.href on a public
    page) -- is shared with the in-app bootstrap.
    """
    js = _wrapped_bootstrap_js(ns)

    _preamble = (
        "  var launch = document.getElementById('{ns}Launch');\n"
        "  var mount = document.getElementById('{ns}Mount');\n"
        "  if (!launch || !mount || launch.__wrapBound) return;\n"
        "  launch.__wrapBound = true;\n"
        "  var loaded = false, loading = false;\n"
        "  // Lets the host page (e.g. the weekly hub's week switcher) point the button\n"
        "  // at a different lazy URL and drop the already-fetched overlay, so the next\n"
        "  // click re-fetches for the new target.\n"
        "  launch.__wrappedReset = function () { loaded = false; mount.innerHTML = ''; };\n"
    ).replace("{ns}", ns)
    assert _preamble in js, "wrapped bootstrap preamble changed; update _wrapped_public_bootstrap_js"
    js = js.replace(
        _preamble,
        "  // Public share page: no launcher; the deck is already in the DOM.\n"
        "  var launch = null;\n"
        "  var mount = null;\n",
    )

    _launch_handler = """  launch.addEventListener('click', function () {
    var url = launch.getAttribute('data-wrapped-url');
    if (loaded || !url) { openWrapped(); return; }   // already injected, or nothing to fetch
    if (loading) return;
    loading = true;
    launch.classList.add('wrapped-launch-loading');
    // The overlay endpoint can 502 while a cold worker is still building the
    // league context, so retry with backoff instead of dying silently.
    var attempts = 0;
    function done() { loading = false; launch.classList.remove('wrapped-launch-loading'); }
    function fail() {
      if (attempts < 3) { setTimeout(tryFetch, attempts * 1500); return; }
      done();
      toast('Could not load the story \\u2014 tap to try again');
    }
    function tryFetch() {
      attempts++;
      fetch(url, { headers: { 'X-Requested-With': 'fetch' } })
        .then(function (r) { if (!r.ok) throw new Error('http ' + r.status); return r.json(); })
        .then(function (d) {
          if (d && d.html) {
            mount.innerHTML = d.html;
            // Snapshot the pristine overlay (before open() mutates it) for the
            // copy-link share flow.
            launch.__wrappedPristine = d.html;
            loaded = true;
            done();
            openWrapped();
          } else { fail(); }
        })
        .catch(function () { fail(); });
    }
    tryFetch();
  });
})();"""
    assert _launch_handler in js, "wrapped bootstrap launch handler changed; update _wrapped_public_bootstrap_js"
    js = js.replace(_launch_handler, "  openWrapped();\n})();")
    return js


def _wrapped_share_og_description(share_data: dict | None) -> str:
    """One-line description for Open Graph unfurls, built from highlights."""
    data = share_data or {}
    bits = []
    for h in (data.get("highlights") or [])[:4]:
        k = str(h.get("k") or "").title()
        n = str(h.get("n") or "")
        v = str(h.get("v") or "")
        bits.append(f"{k}: {n} ({v})" if v else f"{k}: {n}")
    desc = " • ".join(b for b in bits if b.strip(": ()"))
    return desc or "A fantasy football Wrapped story"


def render_wrapped_share_page(*, overlay_html: str, share_data: dict | None,
                              label: str, ns: str, css_url: str,
                              logo_url: str) -> str:
    """Standalone public HTML page for a shared Wrapped deck (no auth).

    The stored overlay markup is injected verbatim and auto-played with the
    public bootstrap variant. noindex keeps league decks out of search; OG
    tags make the link unfurl in iMessage / social apps.
    """
    data = share_data or {}
    title = label or "Fantasy Wrapped"
    desc = _wrapped_share_og_description(data)
    js = _wrapped_public_bootstrap_js(ns)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{_esc(title)}</title>
<meta name="robots" content="noindex, nofollow">
<meta property="og:title" content="{_esc(title, quote=True)}">
<meta property="og:description" content="{_esc(desc, quote=True)}">
<meta property="og:type" content="website">
<meta property="og:image" content="{_esc(logo_url, quote=True)}">
<link rel="stylesheet" href="{_esc(css_url, quote=True)}">
<style>
  html, body {{ margin: 0; padding: 0; background: #0b0b16; }}
  .wrapped-share-cta {{
    position: fixed; bottom: max(18px, env(safe-area-inset-bottom)); left: 0; right: 0;
    z-index: 5000; display: flex; justify-content: center; pointer-events: none;
  }}
  .wrapped-share-cta a {{
    pointer-events: auto; display: inline-flex; align-items: center; gap: 8px;
    padding: 12px 22px; border-radius: 999px; font-weight: 800; font-size: 15px;
    color: #fff; text-decoration: none;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    box-shadow: 0 8px 30px rgba(56, 189, 248, .35);
  }}
</style>
</head>
<body>
{overlay_html}
<div class="wrapped-share-cta"><a href="/">Make your own Wrapped</a></div>
<script>window.__wrappedSharePublic = true;</script>
<script>{js}</script>
</body>
</html>
"""


_WRAPPED_BOOTSTRAP_JS = r"""
(function () {
  var launch = document.getElementById('wrappedLaunch');
  var mount = document.getElementById('wrappedMount');
  if (!launch || !mount || launch.__wrapBound) return;
  launch.__wrapBound = true;
  var loaded = false, loading = false;
  // Lets the host page (e.g. the weekly hub's week switcher) point the button
  // at a different lazy URL and drop the already-fetched overlay, so the next
  // click re-fetches for the new target.
  launch.__wrappedReset = function () { loaded = false; mount.innerHTML = ''; };

  var FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif";

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }
  function ellip(ctx, text, max) {
    text = String(text == null ? '' : text);
    if (ctx.measureText(text).width <= max) return text;
    // Trim by code points (Array.from), not UTF-16 units, so an emoji is never
    // cut in half into a broken glyph.
    var chars = Array.from(text);
    while (chars.length > 1 && ctx.measureText(chars.join('') + '…').width > max) chars.pop();
    return chars.join('') + '…';
  }
  // Self-contained count-up for the public share page, which doesn't load the
  // app bundle that defines window.brCountUp. Same opts shape ({to, dp, suffix, dur}).
  function wrappedCountUp(el, opts) {
    if (!el) return;
    opts = opts || {};
    var to = opts.to;
    if (to == null || isNaN(to)) return;
    var dp = opts.dp != null ? opts.dp : 0;
    var suffix = opts.suffix || '';
    var dur = opts.dur || 1100;
    function fmt(n) { return n.toFixed(dp) + suffix; }
    try {
      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) { el.textContent = fmt(to); return; }
    } catch (e) {}
    var start = null;
    function frame(ts) {
      if (start == null) start = ts;
      var p = Math.min(1, (ts - start) / dur);
      var eased = 1 - Math.pow(1 - p, 3);
      el.textContent = fmt(to * eased);
      if (p < 1) requestAnimationFrame(frame);
      else el.textContent = fmt(to);
    }
    requestAnimationFrame(frame);
  }
  // Centered text drawn via manual measurement with left alignment. WebKit
  // (iOS Safari) misplaces canvas fillText when textAlign is 'center' and the
  // string contains emoji -- the text splits into runs and the alignment is
  // applied wrong -- so team names with emoji drifted off-center. Measuring and
  // positioning ourselves sidesteps that entirely.
  function centerText(ctx, text, cx, y, maxW) {
    var t = maxW ? ellip(ctx, text, maxW) : String(text == null ? '' : text);
    var w = ctx.measureText(t).width;
    var a = ctx.textAlign; ctx.textAlign = 'left';
    ctx.fillText(t, cx - w / 2, y);
    ctx.textAlign = a;
  }
  // Centered text with letter-spacing (px between glyphs).
  function lsCenter(ctx, text, cx, y, ls) {
    var chars = Array.from(String(text == null ? '' : text));
    var i, total = 0;
    for (i = 0; i < chars.length; i++) total += ctx.measureText(chars[i]).width + ls;
    total -= ls;
    var x = cx - total / 2, a = ctx.textAlign; ctx.textAlign = 'left';
    for (i = 0; i < chars.length; i++) { ctx.fillText(chars[i], x, y); x += ctx.measureText(chars[i]).width + ls; }
    ctx.textAlign = a;
  }
  // Measured width of letter-spaced text (matches lsLeft's layout).
  function lsWidth(ctx, text, ls) {
    var chars = Array.from(String(text == null ? '' : text));
    var t = 0;
    for (var i = 0; i < chars.length; i++) t += ctx.measureText(chars[i]).width + ls;
    return Math.max(0, t - ls);
  }
  // Left-aligned text with letter-spacing, returns the ending x.
  function lsLeft(ctx, text, x, y, ls) {
    var chars = Array.from(String(text == null ? '' : text));
    var a = ctx.textAlign; ctx.textAlign = 'left';
    for (var i = 0; i < chars.length; i++) { ctx.fillText(chars[i], x, y); x += ctx.measureText(chars[i]).width + ls; }
    ctx.textAlign = a; return x;
  }
  function drawCrown(ctx, cx, baseY, w, color) {
    var h = w * 0.72, x0 = cx - w / 2, x1 = cx + w / 2;
    ctx.fillStyle = color; ctx.beginPath();
    ctx.moveTo(x0, baseY);
    ctx.lineTo(x0, baseY - h * 0.55);
    ctx.lineTo(x0 + w * 0.26, baseY - h * 0.2);
    ctx.lineTo(cx, baseY - h);
    ctx.lineTo(x1 - w * 0.26, baseY - h * 0.2);
    ctx.lineTo(x1, baseY - h * 0.55);
    ctx.lineTo(x1, baseY);
    ctx.closePath(); ctx.fill();
  }

  // Paint the single shareable 'Season Wrapped' summary card (story format),
  // matching the overlay's editorial style: left-aligned on a strict margin,
  // brand type (Archivo / Inter), an outlined season mark, and a footer bug.
  var F_BODY = "'Archivo', -apple-system, BlinkMacSystemFont, sans-serif";
  var F_DISP = "'InterVariable', 'Archivo', -apple-system, sans-serif";
  function paintWrappedCard(data, logo) {
    var W = 1080, H = 1920, PAD = 90, PURPLE = '#a78bfa', GOLD = '#f5c451';
    var c = document.createElement('canvas'); c.width = W; c.height = H;
    var ctx = c.getContext('2d');
    // Ground: deep purple gradient + one soft glow, top-left.
    var g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, '#221247'); g.addColorStop(0.55, '#120b26'); g.addColorStop(1, '#07060c');
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    var rg = ctx.createRadialGradient(W * 0.2, 480, 0, W * 0.2, 480, 760);
    rg.addColorStop(0, 'rgba(124,58,237,0.30)'); rg.addColorStop(1, 'rgba(124,58,237,0)');
    ctx.fillStyle = rg; ctx.fillRect(0, 0, W, H);
    ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';

    // Outlined season mark bleeding off the top-right. Weekly decks pass
    // data.week, which swaps the mark/kicker/footer to "WEEK N" branding.
    var _wkTag = data.week ? ('WEEK ' + data.week) : null;
    var yr = _wkTag ? ('W' + data.week) : ("'" + String(data.season || '').slice(-2));
    ctx.save();
    ctx.font = '900 330px ' + F_DISP;
    ctx.strokeStyle = PURPLE; ctx.lineWidth = 4; ctx.globalAlpha = 0.13;
    var yw = ctx.measureText(yr).width;
    ctx.strokeText(yr, W - yw + 40, 400);
    ctx.restore();

    // Header: logo, league, kicker.
    if (logo) {
      var lw = 150, lh = lw * (logo.naturalHeight || 454) / (logo.naturalWidth || 512);
      ctx.drawImage(logo, PAD, 130, lw, lh);
    }
    ctx.fillStyle = 'rgba(255,255,255,0.95)'; ctx.font = '800 46px ' + F_BODY;
    ctx.fillText(ellip(ctx, data.league || 'League', W - 2 * PAD), PAD, 388);
    ctx.fillStyle = PURPLE;
    ctx.fillRect(PAD, 432, 54, 8);
    ctx.font = '800 34px ' + F_BODY;
    lsLeft(ctx, _wkTag ? (_wkTag + ' WRAPPED') : 'SEASON WRAPPED', PAD + 76, 444, 8);

    // Champion block.
    var y = 560;
    if (data.champion) {
      ctx.fillStyle = GOLD;
      ctx.fillRect(PAD, y, 54, 8);
      ctx.font = '800 34px ' + F_BODY;
      lsLeft(ctx, (String(data.season || '') + ' CHAMPION').trim(), PAD + 76, y + 12, 8);
      ctx.fillStyle = '#ffffff';
      var cs = 92; ctx.font = '900 ' + cs + 'px ' + F_DISP;
      while (ctx.measureText(data.champion).width > W - 2 * PAD && cs > 54) {
        cs -= 4; ctx.font = '900 ' + cs + 'px ' + F_DISP;
      }
      ctx.fillText(ellip(ctx, data.champion, W - 2 * PAD), PAD, y + 130);
      var ry0 = y + 176;
      if (data.champion_record) {
        ctx.font = '800 32px ' + F_BODY;
        var bw = ctx.measureText(data.champion_record).width + 44;
        var bg = ctx.createLinearGradient(PAD, 0, PAD + bw, 0);
        bg.addColorStop(0, '#f5c451'); bg.addColorStop(1, '#e8a41f');
        roundRect(ctx, PAD, ry0, bw, 58, 10);
        ctx.fillStyle = bg; ctx.fill();
        ctx.fillStyle = '#0d0a04';
        ctx.fillText(data.champion_record, PAD + 22, ry0 + 41);
        ctx.fillStyle = 'rgba(255,255,255,0.66)'; ctx.font = '700 34px ' + F_BODY;
        ctx.fillText('League champion', PAD + bw + 24, ry0 + 41);
      }
      y = ry0 + 120;
    }

    // Divider then highlight rows: label over name, value right in accent.
    ctx.fillStyle = 'rgba(255,255,255,0.12)'; ctx.fillRect(PAD, y, W - 2 * PAD, 2);
    y += 52;
    var hs = (data.highlights || []).slice(0, 5);
    var valX = W - PAD;
    hs.forEach(function (row, i) {
      ctx.fillStyle = PURPLE; ctx.font = '800 28px ' + F_BODY;
      lsLeft(ctx, String(row.k || '').toUpperCase(), PAD, y, 6);
      var vtxt = String(row.v || '');
      ctx.font = '900 48px ' + F_DISP;
      var vw = vtxt ? ctx.measureText(vtxt).width : 0;
      if (vtxt) { ctx.fillStyle = PURPLE; ctx.fillText(vtxt, valX - vw, y + 62); }
      ctx.fillStyle = '#ffffff'; ctx.font = '800 46px ' + F_BODY;
      ctx.fillText(ellip(ctx, row.n, (valX - PAD) - (vw ? vw + 34 : 0)), PAD, y + 62);
      if (i < hs.length - 1) {
        ctx.fillStyle = 'rgba(255,255,255,0.1)';
        ctx.fillRect(PAD, y + 100, W - 2 * PAD, 1);
      }
      y += 152;
    });

    // Footer bug: wordmark + hairline + season.
    var fy = H - 120;
    var fx = PAD;
    if (logo) {
      var flw = 104, flh = flw * (logo.naturalHeight || 454) / (logo.naturalWidth || 512);
      ctx.globalAlpha = 0.85;
      ctx.drawImage(logo, PAD, fy - flh / 2, flw, flh);
      ctx.globalAlpha = 1;
      fx = PAD + flw + 28;
    }
    ctx.font = '700 28px ' + F_BODY;
    var seasonTxt = _wkTag ? _wkTag : ((String(data.season || '') + ' SEASON').trim());
    var stW = lsWidth(ctx, seasonTxt, 6);
    ctx.fillStyle = 'rgba(255,255,255,0.14)';
    ctx.fillRect(fx, fy, (W - PAD - stW - 30) - fx, 1);
    ctx.fillStyle = 'rgba(255,255,255,0.42)';
    lsLeft(ctx, seasonTxt, W - PAD - stW, fy + 10, 6);
    return c;
  }

  // Wire up nav/keyboard/touch on a freshly-injected overlay once, and expose
  // its open() on the element so repeat clicks just reopen it.
  function bindOverlay() {
    var overlay = document.getElementById('wrappedOverlay');
    if (!overlay) return null;
    if (overlay.__navBound) return overlay;
    overlay.__navBound = true;
    var stage = document.getElementById('wrappedStage');
    var slides = Array.prototype.slice.call(stage.querySelectorAll('.wrapped-slide'));
    var bars = Array.prototype.slice.call(overlay.querySelectorAll('.wrapped-bar'));
    var idx = 0, timer = null, DUR = 5000;   // each slide auto-advances after DUR
    var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function clearTimer() { if (timer) { clearTimeout(timer); timer = null; } }
    function scheduleNext(n) {
      clearTimer();
      if (n >= slides.length - 1) return;   // the finale (champion) holds
      timer = setTimeout(function () { go(n + 1); }, DUR);
    }
    // Story-style ticks, driven from JS so they always mirror navigation:
    // skipping ahead snaps the skipped bar full and restarts the next from 0,
    // and going back empties the later bars and replays the revisited one.
    function setBars(n) {
      bars.forEach(function (b, i) {
        var fill = b.firstElementChild;
        if (!fill) return;
        fill.style.transition = 'none';
        fill.style.width = i < n ? '100%' : '0%';
        b.classList.toggle('current', i === n);
      });
      var cur = bars[n] && bars[n].firstElementChild;
      if (!cur) return;
      if (reduce || n >= slides.length - 1) { cur.style.width = '100%'; return; }
      void cur.offsetWidth;   // commit the 0% start before animating
      cur.style.transition = 'width ' + DUR + 'ms linear';
      cur.style.width = '100%';
    }
    // Freeze the current bar in place (share sheet up), then replay it.
    function freezeBar() {
      var cur = bars[idx] && bars[idx].firstElementChild;
      if (!cur) return;
      var w = cur.getBoundingClientRect().width;
      cur.style.transition = 'none';
      cur.style.width = w + 'px';
    }
    function playSlide(n) {
      slides.forEach(function (s, i) {
        s.classList.toggle('active', i === n);
        if (i < n) s.classList.add('seen'); else s.classList.remove('seen');
      });
      setBars(n);
      var el = slides[n];
      if (!el) return;
      // Tint the current progress tick with the active slide's accent.
      try { overlay.style.setProperty('--wa', getComputedStyle(el).getPropertyValue('--wa')); } catch (e) {}
      var num = el.querySelector('[data-w-count]');
      if (num) {
        (window.brCountUp || wrappedCountUp)(num, { to: parseFloat(num.getAttribute('data-w-count')),
          dp: parseInt(num.getAttribute('data-w-dp') || '0', 10),
          suffix: num.getAttribute('data-w-suffix') || '', dur: 1100 });
      }
      if (el.getAttribute('data-kind') === 'champion' && window.brConfetti) {
        setTimeout(function () { window.brConfetti(el, { palette: ['#f5c451','#e0a828','#fff1c2','#ffffff'], y: el.clientHeight * 0.4, count: 120 }); }, 350);
      }
      scheduleNext(n);
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
      clearTimer();
      overlay.hidden = true; overlay.setAttribute('aria-hidden', 'true');
      document.documentElement.style.overflow = '';
    }
    overlay.__open = open;
    document.getElementById('wrappedClose').addEventListener('click', close);
    document.getElementById('wrappedNext').addEventListener('click', function () { go(idx + 1); });
    document.getElementById('wrappedPrev').addEventListener('click', function () { go(idx - 1); });
    document.addEventListener('keydown', function (e) {
      if (overlay.hidden) return;
      if (e.key === 'Escape') close();
      else if (e.key === 'ArrowRight' || e.key === ' ') { e.preventDefault(); go(idx + 1); }
      else if (e.key === 'ArrowLeft') go(idx - 1);
    });
    var sx = null;
    stage.addEventListener('touchstart', function (e) { sx = e.touches[0].clientX; }, { passive: true });
    stage.addEventListener('touchend', function (e) {
      if (sx === null) return;
      var dx = e.changedTouches[0].clientX - sx; sx = null;
      if (Math.abs(dx) > 40) go(idx + (dx < 0 ? 1 : -1));
    }, { passive: true });

    // Share: paint the summary card and hand it to the native share sheet.
    var shareBtn = document.getElementById('wrappedShare');
    if (shareBtn) {
      if (window.brBrandLogo) window.brBrandLogo();   // pre-warm the wordmark
      shareBtn.addEventListener('click', function () {
        clearTimer();   // pause auto-advance while the share sheet is up
        freezeBar();    // hold the tick where it was
        var data = {};
        try { data = JSON.parse((document.getElementById('wrappedShareData') || {}).textContent || '{}'); } catch (e) {}
        shareBtn.classList.add('wrapped-share-busy');
        var done = function () {
          shareBtn.classList.remove('wrapped-share-busy');
          // Replay the slide's clock from the top: tick and timer back in sync.
          setBars(idx); scheduleNext(idx);
        };
        var logoP = window.brBrandLogo ? window.brBrandLogo() : Promise.resolve(null);
        logoP.then(function (logo) {
          var canvas;
          try { canvas = paintWrappedCard(data, logo); } catch (e) { done(); return; }
          if (window.brShareCanvas) {
            var _fname = data.week ? ('week-' + data.week + '-wrapped.png') : 'season-wrapped.png';
            var _ttl = (data.league || 'League') + ' - '
              + (data.week ? ('Week ' + data.week + ' Wrapped') : 'Season Wrapped');
            window.brShareCanvas(canvas, _fname, _ttl).then(done, done);
          } else {
            try { window.open(canvas.toDataURL('image/png'), '_blank'); } catch (e) {}
            done();
          }
        });
      });
    }
    // Copy-link: mint a public shareable URL for this deck. The pristine
    // overlay HTML is snapshotted at inject time (before open() mutates it),
    // so the shared page always starts clean on slide one.
    var linkBtn = document.getElementById('wrappedLink');
    if (linkBtn) {
      // The button id is namespaced ("weekly-wrappedLink"); strip the suffix
      // to recover the ns for the POST body.
      var linkNs = linkBtn.id.replace(/Link$/, '');
      linkBtn.addEventListener('click', function () {
        // Public shared page: the address bar already IS the share link.
        if (window.__wrappedSharePublic || !launch) {
          copyText(window.location.href, function (ok) {
            toast(ok ? 'Link copied' : 'Could not copy link');
          });
          return;
        }
        var pristine = launch.__wrappedPristine;
        if (!pristine) {
          toast('Open the story first');
          return;
        }
        var data = {};
        try { data = JSON.parse((document.getElementById('wrappedShareData') || {}).textContent || '{}'); } catch (e) {}
        linkBtn.classList.add('wrapped-share-busy');
        fetch('/api/wrapped/share', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            kind: data.week ? 'weekly' : 'season',
            ns: linkNs,
            overlay_html: pristine,
            share_data: data
          })
        })
          .then(function (r) { return r.json(); })
          .then(function (d) {
            linkBtn.classList.remove('wrapped-share-busy');
            if (d && d.url) {
              copyText(d.url, function (ok) {
                toast(ok ? 'Link copied — anyone with the link can view'
                         : 'Could not copy link');
              });
            } else { toast('Could not create link'); }
          })
          .catch(function () {
            linkBtn.classList.remove('wrapped-share-busy');
            toast('Could not create link');
          });
      });
    }
    return overlay;
  }

  // Clipboard helper with a textarea fallback for older iOS WebViews.
  function copyText(text, cb) {
    function fin(ok) { try { cb(!!ok); } catch (e) {} }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () { fin(true); }, function () { fin(false); });
      return;
    }
    try {
      var ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      var ok = false;
      try { ok = document.execCommand('copy'); } catch (e) {}
      document.body.removeChild(ta);
      fin(ok);
    } catch (e) { fin(false); }
  }

  // Small toast anchored to the overlay.
  function toast(msg) {
    var t = document.getElementById('wrappedToast');
    if (!t) {
      t = document.createElement('div');
      t.id = 'wrappedToast';
      t.className = 'wrapped-toast';
      t.setAttribute('role', 'status');
      document.body.appendChild(t);
    }
    t.textContent = msg;
    t.classList.add('show');
    if (t.__tm) clearTimeout(t.__tm);
    t.__tm = setTimeout(function () { t.classList.remove('show'); }, 2600);
  }

  function openWrapped() {
    var overlay = bindOverlay();
    if (overlay && overlay.__open) overlay.__open();
  }

  launch.addEventListener('click', function () {
    var url = launch.getAttribute('data-wrapped-url');
    if (loaded || !url) { openWrapped(); return; }   // already injected, or nothing to fetch
    if (loading) return;
    loading = true;
    launch.classList.add('wrapped-launch-loading');
    // The overlay endpoint can 502 while a cold worker is still building the
    // league context, so retry with backoff instead of dying silently.
    var attempts = 0;
    function done() { loading = false; launch.classList.remove('wrapped-launch-loading'); }
    function fail() {
      if (attempts < 3) { setTimeout(tryFetch, attempts * 1500); return; }
      done();
      toast('Could not load the story \u2014 tap to try again');
    }
    function tryFetch() {
      attempts++;
      fetch(url, { headers: { 'X-Requested-With': 'fetch' } })
        .then(function (r) { if (!r.ok) throw new Error('http ' + r.status); return r.json(); })
        .then(function (d) {
          if (d && d.html) {
            mount.innerHTML = d.html;
            // Snapshot the pristine overlay (before open() mutates it) for the
            // copy-link share flow.
            launch.__wrappedPristine = d.html;
            loaded = true;
            done();
            openWrapped();
          } else { fail(); }
        })
        .catch(function () { fail(); });
    }
    tryFetch();
  });
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
            "league_pages.page_history",
            platform=base_platform,
            season=base_season,
            league_id=base_league_id,
            history_season=yr,
        )
        selected = "selected" if yr == selected_history_season else ""
        options_html.append(f"<option value='{href}' {selected}>{yr}</option>")

    league_name = league.get("name") or "League History"

    _partial_chip = ""
    try:
        _pw = _playoff_start_week(league)
        if df_weekly is not None and not getattr(df_weekly, "empty", True) and "week" in df_weekly.columns:
            _max_w = int(pd.to_numeric(df_weekly["week"], errors="coerce").max() or 0)
            if 0 < _max_w < (_pw - 1):
                _partial_chip = (
                    f'<span class="history-partial-chip" style="display:inline-block;margin-top:8px;'
                    f'padding:3px 8px;border-radius:999px;font-size:11px;font-weight:700;'
                    f'background:color-mix(in srgb,#b45309 14%,transparent);color:#b45309;">'
                    f'Partial season · through week {_max_w}</span>'
                )
    except Exception:
        _partial_chip = ""

    # Season Wrapped: a stories-style recap, lazy-loaded on first click so its
    # per-week boxscore fetches never block the page render. Here we only build
    # the cheap (no-boxscore) slide set to decide whether to show the launcher.
    _wrapped_cheap = _build_wrapped_slides(
        history_ctx, summary, league_name, selected_history_season, include_players=False
    )
    if len(_wrapped_cheap) >= 3:
        _wrapped_url = (
            f"/api/history/{base_platform}/{base_season}/{base_league_id}"
            f"/wrapped?history_season={selected_history_season}"
        )
        _wrapped_btn = _wrapped_launcher_html(_wrapped_url)
    else:
        _wrapped_btn = ""
    _wrapped_overlay = ""  # injected lazily into #wrappedMount by the launcher

    # Shimmer skeletons shaped like the content they stand in for, so the lazy
    # /api/history/* sections read as "arriving" rather than a bare spinner.
    # (Unused when prerendered sections are provided.)
    _sk_card = (
        "<div class='history-card'>"
        "<div class='sk-shimmer sk-line sk-line--sm' style='width:45%'></div>"
        "<div class='sk-shimmer sk-line sk-line--lg' style='width:72%;margin-top:8px'></div>"
        "<div class='sk-shimmer sk-line sk-line--sm' style='width:58%;margin-top:6px'></div>"
        "</div>"
    )
    awards_skeleton = f"<div class='history-awards-grid'>{_sk_card * 6}</div>"
    _sk_row = (
        "<div style='display:flex;align-items:center;gap:12px;padding:10px 0;'>"
        "<div class='sk-shimmer sk-avatar' style='width:22px;height:22px'></div>"
        "<div class='sk-shimmer sk-line' style='width:38%;margin:0'></div>"
        "<div class='sk-shimmer sk-line' style='width:13%;margin:0 0 0 auto'></div>"
        "<div class='sk-shimmer sk-line' style='width:13%;margin:0'></div>"
        "</div>"
    )
    standings_skeleton = f"<div style='padding-top:12px'>{_sk_row * 8}</div>"
    chart_skeleton = (
        "<div class='sk-shimmer' style='width:100%;height:280px;"
        "border-radius:12px;margin-top:6px'></div>"
    )
    awards_html    = prerendered["summary"]   if prerendered else awards_skeleton
    standings_html = prerendered["standings"] if prerendered else standings_skeleton
    chart_html     = prerendered["chart"]     if prerendered else chart_skeleton
    tour_input     = '<input type="hidden" id="historyTourMode" value="1">' if prerendered else ""

    bracket_html = _get_history_bracket_html(history_ctx)

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
          <div class="history-kicker">BR Fantasy · League History</div>
          <h1 class="history-title">
            <span class="history-title-accent">{_esc(league_name)}</span> • {selected_history_season}
          </h1>
          <p class="history-subtitle">{_esc(recap_line)}</p>
          {_partial_chip}
        </div>

        <div style="display:flex;flex-direction:column;align-items:flex-end;gap:12px;">
          <div class="history-header-actions">
            {_wrapped_btn}
            <a href="/{base_platform}/{base_season}/{base_league_id}/awards" class="awards-page-nav-link">
              <i class="fa-solid fa-trophy"></i>
              All-Time Awards
            </a>
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
      </div>

      <div class="history-top-grid">
        <div class="card history-awards-panel">
          <div class="card-header"><h2>Season Awards</h2></div>
          <div class="card-body" id="historyAwardsContent">
            {awards_html}
          </div>
        </div>

        <div class="card history-standings-panel">
          <div class="card-tabs" data-card="history-standings">
            <div class="tab-strip">
              <button class="tab-btn active" data-tab="standings">Standings</button>
              {'<button class="tab-btn" data-tab="bracket">Playoff Bracket</button>' if bracket_html else ''}
            </div>
            <div class="tab-panels" style="padding:0;">
              <div class="tab-panel active" data-tab="standings" id="historyStandingsContent" style="padding-top:0;">
                {standings_html}
              </div>
              {'<div class="tab-panel" data-tab="bracket">' + bracket_html + '</div>' if bracket_html else ''}
            </div>
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


def build_tour_mock_history_ctx(df_weekly) -> dict:
    """History context for the tour/demo, from a pre-built mock weekly frame.

    Extracted from app.py so the /history route can stay a thin blueprint handler;
    the caller supplies the mock weekly DataFrame (app._build_tour_mock_df_weekly)."""
    return {
        "platform": "sleeper",
        "season": 2024,
        "league_id": "tour_mock",
        "resolved_league_id": "tour_mock",
        "league": {
            "name": "Demo League",
            "league_id": "tour_mock",
            "settings": {"playoff_week_start": 14},
        },
        "df_weekly": df_weekly,
        "roster_map": {},
        "users": {},
        "rosters": [],
        "offseason_mode": False,
    }
