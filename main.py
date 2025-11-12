
import argparse
from dashboard_services.api import get_users, get_rosters, get_matchups, get_nfl_state
from dashboard_services.players import get_players_map, build_roster_map
from dashboard_services.injuries import build_injury_report, render_injury_accordion
from dashboard_services.matchups import build_matchup_preview, render_matchup_carousel_weeks
from dashboard_services.pdf_theme import apply_theme, build_pdf

def main():
    ap = argparse.ArgumentParser(description="League Blueprint - modular demo")
    ap.add_argument("--league", required=True, help="Sleeper league id")
    ap.add_argument("--weeks", type=int, default=get_nfl_state().get("week"), help="Which week to preview")
    args = ap.parse_args()

    league_id = args.league
    week = args.weeks

    players_map = get_players_map()
    roster_map = build_roster_map(league_id)

    inj_df = build_injury_report(league_id)
    injury_html = render_injury_accordion(inj_df)

    matchups = build_matchup_preview(league_id, week, roster_map, players_map)
    slides_html = "".join(
        ["<div class='m-slide'>" + m['left']['name'] + " vs " + m['right']['name'] + "</div>"
         for m in matchups]
    )
    carousel_html = render_matchup_carousel_weeks({week: slides_html}, week)

    print("injury_html length:", len(injury_html))
    print("carousel_html length:", len(carousel_html))

if __name__ == "__main__":
    main()
