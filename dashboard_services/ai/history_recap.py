from __future__ import annotations

import json
import threading

from dashboard_services.ai.cache import load_cached_ai_text, save_cached_ai_text
from dashboard_services.ai.prompts import get_ai_client
from dashboard_services.ai.renderer import ai_available
from dashboard_services.platform_api import get_bracket


def build_history_recap_payload(history_ctx: dict, roster_id: str) -> dict:
    """Build structured payload for AI season recap analysis."""
    league = history_ctx.get("league", {})
    df_weekly = history_ctx.get("df_weekly")
    summary = history_ctx.get("summary", {})

    # Find team data
    team_name = "Unknown Team"
    team_stats = {}

    if df_weekly is not None and not df_weekly.empty:
        # Filter for the specific roster_id
        team_df = df_weekly[df_weekly["roster_id"] == roster_id]

        if not team_df.empty:
            # Get team name from first row
            team_name = team_df.iloc[0].get("owner", "Unknown Team")

            # Get actual matchup results for real win/loss calculation
            actual_record = _get_actual_record(team_df, history_ctx, roster_id)

            # Get playoff results
            playoff_results = _get_playoff_results(history_ctx, roster_id)

            # Calculate team stats
            team_stats = {
                "record": actual_record,
                "playoff_finish": playoff_results.get("finish", "Did not make playoffs"),
                "made_playoffs": playoff_results.get("made_playoffs", False),
                "points_for": float(team_df["points"].sum()),
                "points_against": float(team_df["points_against"].sum()) if "points_against" in team_df.columns else 0,
                "best_week": _get_best_week(team_df),
                "worst_week": _get_worst_week(team_df),
                "weekly_scores": team_df["points"].tolist(),
                "final_standing": _get_final_standing(team_df, league),
            }

    # League context
    league_context = {
        "champion": summary.get("champion", "Unknown"),
        "runner_up": summary.get("runner_up", "Unknown"),
        "top_scorer": summary.get("top_scorer_team", "Unknown"),
        "total_teams": len(league.get("rosters", [])),
        "playoff_teams": _get_playoff_teams_count(league),
    }

    return {
        "team": {
            "name": team_name,
            "roster_id": roster_id,
            **team_stats,
        },
        "league": league_context,
        "season": league.get("season", "Unknown"),
        "summary": summary,
    }


def _get_playoff_results(history_ctx: dict, roster_id: str) -> dict:
    """Get playoff results from bracket data."""
    league = history_ctx.get("league", {})
    season = history_ctx.get("season")
    platform = history_ctx.get("platform", "sleeper")
    league_id = history_ctx.get("league_id")

    if not all([season, league_id]):
        return {"made_playoffs": False, "finish": "Did not make playoffs"}

    try:
        # Get league settings to determine playoff start
        settings = league.get("settings", {})
        playoff_week_start = int(settings.get("playoff_week_start") or 0)

        if not playoff_week_start:
            return {"made_playoffs": False, "finish": "No playoffs in this league"}

        # Get bracket data
        winners_bracket = get_bracket(platform, league_id, "winners", season) or []
        losers_bracket = get_bracket(platform, league_id, "losers", season) or []

        # Search for this team in brackets
        playoff_result = None
        max_round = 0

        # Process Sleeper bracket format: [{r: 1, m: 1, t1: 3, t2: 6, w: null, l: null}, ...]
        def process_sleeper_bracket(bracket_data, bracket_name):
            nonlocal playoff_result, max_round

            for matchup in bracket_data:
                if not isinstance(matchup, dict):
                    continue

                round_num = matchup.get("r", 0)
                match_id = matchup.get("m", 0)
                t1 = matchup.get("t1")
                t2 = matchup.get("t2")
                winner = matchup.get("w")
                loser = matchup.get("l")

                max_round = max(max_round, round_num)

                # Check if team is directly in this matchup
                team_in_matchup = False
                if t1 is not None:
                    if isinstance(t1, dict):
                        # Handle {w: 1} or {l: 1} format
                        if "w" in t1 or "l" in t1:
                            pass  # Team comes from previous match, will check later
                        elif str(t1) == str(roster_id):
                            team_in_matchup = True
                    elif str(t1) == str(roster_id):
                        team_in_matchup = True

                if t2 is not None and not team_in_matchup:
                    if isinstance(t2, dict):
                        # Handle {w: 1} or {l: 1} format
                        if "w" in t2 or "l" in t2:
                            pass  # Team comes from previous match, will check later
                        elif str(t2) == str(roster_id):
                            team_in_matchup = True
                    elif str(t2) == str(roster_id):
                        team_in_matchup = True

                # Check if team won or lost this matchup
                team_won = False
                team_lost = False
                if winner and str(winner) == str(roster_id):
                    team_won = True
                elif loser and str(loser) == str(roster_id):
                    team_lost = True

                if team_in_matchup or team_won or team_lost:
                    if not playoff_result:
                        playoff_result = {"made_playoffs": True, "round_reached": round_num}

                    # Determine finish based on round and result
                    if bracket_name == "winners":
                        total_rounds = max_round
                        if round_num == total_rounds:
                            if team_won:
                                playoff_result["finish"] = "League Champion"
                            elif team_lost:
                                playoff_result["finish"] = "Championship Runner-up"
                        elif round_num == total_rounds - 1:
                            playoff_result["finish"] = f"Semifinalist (Round {round_num})"
                        else:
                            playoff_result["finish"] = f"Playoffs - Round {round_num}"
                    else:
                        playoff_result["finish"] = f"Loser's Bracket - Round {round_num}"

        # Process winners bracket
        process_sleeper_bracket(winners_bracket, "winners")

        # Process losers bracket if no result found
        if not playoff_result:
            process_sleeper_bracket(losers_bracket, "losers")

        # If team made playoffs but no specific finish, default to first round
        if playoff_result and "finish" not in playoff_result:
            playoff_result["finish"] = "Playoffs - First Round"

        return playoff_result or {"made_playoffs": False, "finish": "Did not make playoffs"}

    except Exception as e:
        return {"made_playoffs": False, "finish": "Playoff data unavailable"}


def _get_actual_record(team_df, history_ctx: dict, roster_id: str) -> str:
    """Get actual win-loss record using the same logic as service.py."""
    # df_weekly is directly in the context
    df_weekly = history_ctx.get("df_weekly")

    if df_weekly is None or df_weekly.empty:
        return "0-0"

    # Filter for this team
    team_weekly = df_weekly[df_weekly["roster_id"] == roster_id]

    if team_weekly.empty:
        return "0-0"

    wins = 0
    losses = 0
    ties = 0

    # Group by week and matchup_id to find opponents (same as _weekly_results_from_df)
    for (_, matchup_id), group in df_weekly.groupby(["week", "matchup_id"]):
        group = group.sort_values("roster_id")
        if len(group) != 2:
            continue

        a, b = group.iloc[0], group.iloc[1]
        pa, pb = float(a.get("points", 0.0)), float(b.get("points", 0.0))

        # Check if this team is in this matchup
        team_in_matchup = False
        team_result = None

        if str(a.get("roster_id")) == str(roster_id):
            team_in_matchup = True
            if pa > pb:
                team_result = "W"
            elif pb > pa:
                team_result = "L"
            else:
                team_result = "T"
        elif str(b.get("roster_id")) == str(roster_id):
            team_in_matchup = True
            if pb > pa:
                team_result = "W"
            elif pa > pb:
                team_result = "L"
            else:
                team_result = "T"

        if team_in_matchup and team_result:
            if team_result == "W":
                wins += 1
            elif team_result == "L":
                losses += 1
            else:
                ties += 1

    if ties > 0:
        return f"{wins}-{losses}-{ties}"
    return f"{wins}-{losses}"


def _calculate_record(team_df) -> str:
    """Calculate win-loss record from weekly data."""
    # Since we don't have actual win/loss results in the weekly data,
    # we'll provide a more conservative estimate or skip record entirely
    total_games = len(team_df)
    if total_games == 0:
        return "0-0"

    # For now, let's use a more conservative estimate based on top 40% of scores
    # This is still an approximation since we don't have actual matchup results
    points = team_df["points"].sort_values(ascending=False)
    wins_threshold = int(total_games * 0.4)  # Top 40% as wins
    wins = wins_threshold if wins_threshold > 0 else 0
    losses = total_games - wins

    return f"{wins}-{losses}"


def _get_best_week(team_df) -> dict:
    """Get best week performance."""
    best_row = team_df.loc[team_df["points"].idxmax()]
    return {
        "week": int(best_row["week"]),
        "score": float(best_row["points"]),
        "opponent": "Unknown",  # Not available in current data structure
    }


def _get_worst_week(team_df) -> dict:
    """Get worst week performance."""
    worst_row = team_df.loc[team_df["points"].idxmin()]
    return {
        "week": int(worst_row["week"]),
        "score": float(worst_row["points"]),
        "opponent": "Unknown",  # Not available in current data structure
    }


def _get_final_standing(team_df, league: dict) -> int:
    """Get final standing position."""
    # This is a simplified version - you may need to adjust based on your data structure
    standings = league.get("standings", [])
    roster_id = team_df.iloc[0].get("roster_id")
    for i, standing in enumerate(standings, 1):
        if standing.get("roster_id") == roster_id:
            return i
    return len(standings)  # Default to last if not found


def _get_playoff_teams_count(league: dict) -> int:
    """Get number of playoff teams."""
    settings = league.get("settings", {})
    return int(settings.get("playoff_teams", 6))


def get_history_ai_recap(history_ctx: dict, roster_id: str) -> str:
    """Generate AI-powered season recap for a specific team."""
    league = history_ctx.get('league', {})
    season = history_ctx.get('season', 'unknown')
    league_id = league.get('league_id', 'unknown')
    cache_key = f"history_recap_{league_id}_{season}_{roster_id}"

    # Try to get from cache first
    cached = load_cached_ai_text(cache_key)
    if cached:
        return cached

    if not ai_available():
        return _fallback_recap(history_ctx, roster_id)

    try:
        payload = build_history_recap_payload(history_ctx, roster_id)

        # Generate AI result
        result = _generate_recap_ai_result(payload)
        html_out = _render_recap_html(result)

        # Cache the result
        save_cached_ai_text(cache_key, html_out)
        return html_out

    except Exception as e:
        print(f"[history-recap] AI error: {e}")
        return _fallback_recap(history_ctx, roster_id)


def _generate_recap_ai_result(payload: dict) -> dict:
    """Generate AI result for season recap."""
    client = get_ai_client()

    schema = {
        "type": "object",
        "properties": {
            "recap": {
                "type": "string",
                "description": "The season recap text",
            },
            "highlights": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key highlights from the season",
            },
            "grade": {
                "type": "string",
                "enum": ["A", "B", "C", "D", "F"],
                "description": "Overall season grade",
            },
        },
        "required": ["recap", "highlights", "grade"],
        "additionalProperties": False,
    }

    system_prompt = """
    You are a professional fantasy football analyst writing a season recap for a specific team.
    
    Write an engaging, narrative recap from the team's perspective. Your recap should:
    
    1. Be insightful and specific to THIS team's performance
    2. Reference actual stats and data provided
    3. Tell a story - not just list facts
    4. Include key turning points, strengths, and weaknesses
    5. End with a forward-looking perspective
    
    Style guidelines:
    - Write like The Athletic or ESPN analyst
    - 2-4 paragraphs maximum
    - Use strong topic sentences
    - Vary sentence structure
    - Avoid clichés and generic phrases
    - Focus on what actually happened, not what could have happened
    
    Do NOT:
    - Invent facts not in the data
    - Use markdown formatting
    - Include bullet points
    - Repeat the same information
    - Sound like a generic template
    
    Make this feel like a personalized, professional analysis of this team's season.
    """.strip()

    user_prompt = f"""
    Write a season recap for this team:
    
    Team: {payload['team']['name']}
    Full Season including Playoffs if made, Toilet Bowl if not Record: {payload['team'].get('record', 'Unknown')}
    Playoff Finish: {payload['team'].get('playoff_finish', 'Did not make playoffs')}
    Points For: {payload['team'].get('points_for', 0):.1f}
    Points Against: {payload['team'].get('points_against', 0):.1f}
    Final Standing: {payload['team'].get('final_standing', 'Unknown')} of {payload['league']['total_teams']}
    Personal Best Week: Week {payload['team'].get('best_week', {}).get('week', 'N/A')} ({payload['team'].get('best_week', {}).get('score', 0):.1f} points)
    Personal Worst Week: Week {payload['team'].get('worst_week', {}).get('week', 'N/A')} ({payload['team'].get('worst_week', {}).get('score', 0):.1f} points)
    
    League Champion: {payload['league']['champion']}
    Top Scorer: {payload['league']['top_scorer']}
    
    Season: {payload['season']}
    
    Write a compelling recap that captures the essence of this team's season journey, including both regular season and playoff performance.
    """.strip()

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "season_recap",
                "schema": schema,
            }
        },
    )

    data = json.loads(resp.output_text.strip())
    return data


def _render_recap_html(result: dict) -> str:
    """Render AI result as HTML."""
    recap = result.get("recap", "").strip()
    highlights = result.get("highlights", [])
    grade = result.get("grade", "C")

    # Convert recap paragraphs to HTML
    paragraphs = recap.split("\n\n")
    recap_html = "\n".join(f"<p>{p.strip()}</p>" for p in paragraphs if p.strip())

    highlights_html = ""
    if highlights:
        highlights_html = f"""
        <div class="recap-highlights">
            <h4>Key Moments</h4>
            <ul>
                {''.join(f"<li>{highlight}</li>" for highlight in highlights[:3])}
            </ul>
        </div>
        """

    return f"""
    <div class="ai-recap">
        <div class="recap-header">
            <span class="recap-grade grade-{grade.lower()}">Grade: {grade}</span>
        </div>
        {recap_html}
        {highlights_html}
    </div>
    """


def _fallback_recap(history_ctx: dict, roster_id: str) -> str:
    """Fallback recap when AI is unavailable."""
    summary = history_ctx.get("summary", {})
    team_name = "Your Team"

    return f"""
    <div class="ai-recap">
        <div class="recap-header">
            <span class="recap-grade grade-c">Grade: C</span>
        </div>
        <p>Season recap for {team_name} is currently unavailable. The AI analysis system is temporarily down, but you can review your season performance through the standings and statistics above.</p>
        <p>Check back later for a detailed analysis of your season journey, key moments, and performance insights.</p>
    </div>
    """


def get_league_season_summary(history_ctx: dict, season: int) -> str:
    """Return cached AI subtitle immediately; trigger generation in background if missing."""
    league = history_ctx.get("league", {})
    league_id = league.get('league_id', 'unknown')
    cache_key = f"league_summary_{league_id}_{season}"

    cached = load_cached_ai_text(cache_key)
    if cached:
        return cached

    fallback = _fallback_league_summary(history_ctx, season)

    if not ai_available():
        return fallback

    # Generate in background so the page renders immediately with fallback
    def _bg_generate(ctx_snapshot, cache_key_inner, season_inner):
        try:
            summary = ctx_snapshot.get("summary", {})
            payload = _build_league_summary_payload(ctx_snapshot, season_inner)
            result = _generate_league_summary_ai(payload)
            save_cached_ai_text(cache_key_inner, result)
        except Exception:
            pass

    threading.Thread(
        target=_bg_generate,
        args=(history_ctx, cache_key, season),
        daemon=True,
    ).start()

    return fallback


def _build_league_summary_payload(history_ctx: dict, season: int) -> dict:
    league = history_ctx.get("league", {})
    summary = history_ctx.get("summary", {})
    return {
        "season": season,
        "champion": summary.get("champion", "Unknown"),
        "champion_record": summary.get("champion_record", ""),
        "runner_up": summary.get("runner_up", "Unknown"),
        "runner_up_record": summary.get("runner_up_record", ""),
        "top_scorer": summary.get("top_scorer_team", "Unknown"),
        "top_scorer_total": summary.get("top_scorer_value", 0),
        "best_defense": summary.get("best_defense_team", "Unknown"),
        "highest_week": summary.get("highest_week_value", 0),
        "highest_week_team": summary.get("highest_week_team", "Unknown"),
        "lowest_week": summary.get("lowest_week_value", 0),
        "closest_margin": summary.get("closest_margin", 0),
        "biggest_blowout_margin": summary.get("biggest_blowout_margin", 0),
        "unluckiest_team": summary.get("unluckiest_team", "Unknown"),
        "unluckiest_delta": summary.get("unluckiest_delta", 0),
        "league_name": league.get("name", "League"),
        "total_teams": len(league.get("rosters", [])),
    }


def _generate_league_summary_ai(payload: dict) -> str:
    """Generate AI league season summary."""
    client = get_ai_client()

    system_prompt = """
    You are a professional fantasy football analyst writing a compelling season summary.

    Write a 1-2 sentence season recap that captures the essence of the league's story.

    Guidelines:
    - Make it narrative and engaging, not just facts
    - Focus on the most compelling storyline
    - Use active voice and strong verbs
    - Reference actual team names and stats
    - Keep it under 200 characters if possible, max 300
    - No markdown, no bullet points
    - Sound like an ESPN headline analyst

    This will appear as the subtitle on the history page, so make it punchy and memorable.
    """.strip()

    user_prompt = f"""
    Write a compelling season summary based on this data:

    Champion: {payload['champion']} ({payload['champion_record']})
    Runner-up: {payload['runner_up']} ({payload['runner_up_record']})
    Top Scorer: {payload['top_scorer']} ({payload['top_scorer_total']:.1f} total points)
    Best Defense: {payload['best_defense']}
    Highest Week: {payload['highest_week']:.1f} by {payload['highest_week_team']}
    Unluckiest Team: {payload['unluckiest_team']} (finished {payload['unluckiest_delta']} spots below scoring rank)
    Closest Game: {payload['closest_margin']:.1f} point margin
    Biggest Blowout: {payload['biggest_blowout_margin']:.1f} point margin

    League: {payload['league_name']}
    Season: {payload['season']}
    Total Teams: {payload['total_teams']}

    Focus on what made this season memorable and unique.
    """.strip()

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return resp.output_text.strip()


def _fallback_league_summary(history_ctx: dict, season: int) -> str:
    """Fallback league summary when AI is unavailable."""
    summary = history_ctx.get("summary", {})
    champ = summary.get("champion", "Unknown")
    runner = summary.get("runner_up", "Unknown")
    scoring_leader = summary.get("top_scorer_team", "Unknown")
    unlucky = summary.get("unluckiest_team", "Unknown")

    parts = []
    if champ != "—" and runner != "—":
        parts.append(f"{champ} won the {season} title over {runner}.")
    elif champ != "—":
        parts.append(f"{champ} finished as the {season} champion.")

    if scoring_leader != "—":
        parts.append(f"{scoring_leader} led the league in total points.")

    if unlucky != "—" and summary.get("unluckiest_delta", 0) > 0:
        parts.append(
            f"{unlucky} was the rough-luck team, finishing {summary['unluckiest_delta']} spots below its PF rank."
        )

    return " ".join(parts) or f"Review the biggest outcomes and trends from the {season} season."
