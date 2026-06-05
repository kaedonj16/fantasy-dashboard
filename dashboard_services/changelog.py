"""
Changelog entries for the fantasy dashboard.
Each entry represents a user-facing change.
"""

CHANGELOG = [
    {
        "date": "2026-06-05",
        "tag": "new",
        "text": "Playoff Impact: The trade calculator now shows how any deal shifts your playoff odds, projected wins, PPG, top-3 draft pick odds, roster age, and prime years left. All stats are simulated live as you build the trade.",
        "link": "/trade"
    },
    {
        "date": "2026-06-02",
        "tag": "feature",
        "text": "Trade Suggestions: The Win % and playoff-odds pills now reflect the true net effect of a deal. Each package runs a full playoff simulation of your roster after the trade, accounting for both the players you send away and the one you get back, so the odds shift you see is what actually happens to your team, not just the value of the player added.",
        "link": "/trade?tab=suggestions"
    },
    {
        "date": "2026-06-02",
        "tag": "feature",
        "text": "Trade Suggestions: The Consolidate strategy now only surfaces genuine trade-up packages, two or three assets for one difference-maker, and never a 1-for-1 swap.",
        "link": "/trade?tab=suggestions"
    },
    {
        "date": "2026-05-29",
        "tag": "new",
        "text": "Trade Strategy: Archetype-driven suggestions for all four team profiles: Contending, Rebuilding, Consolidate, and Distribute. Each package shows exactly what leaves, what comes back, and how the deal shifts your Win %, playoff odds, and championship probability.",
        "link": "/trade?tab=suggestions"
    },
    {
        "date": "2026-05-28",
        "tag": "new",
        "text": "Schedule Assistant: See matchup difficulty for every player on your roster across any week range. Add or remove players, pick a window, and see how that defense stacks up against that position. Find it under the Players menu.",
        "link": "/schedule"
    },
    {
        "date": "2026-05-24",
        "tag": "new",
        "text": "Breakout Engine: Each candidate now includes historical peer comps drawn from genuine breakout seasons, vacated opportunity totals, and a confidence-adjusted projected PPG range.",
        "link": "/breakouts"
    },
    {
        "date": "2026-05-19",
        "tag": "new",
        "text": "Trade Suggestions: Trade Intel now suggests real trade packages adapted to your roster. See what players have actually been traded for your target, filter out untouchables, and load any package into the Trade Calculator in one click.",
        "link": "/trade?tab=suggestions"
    },
    {
        "date": "2026-05-15",
        "tag": "new",
        "text": "Rookie Draft Assistant: New Draft Board tab on the Prospects page. Analyzes your positional needs vs. the league, recommends 1–2 prospects per pick, shows ADP and grade per row, and tracks who you've drafted in the current session.",
        "link": "/prospects"
    },
    {
        "date": "2026-05-14",
        "tag": "new",
        "text": "Player Search: Search any player from the nav bar. Click the magnifying glass (or press Ctrl+K), type a name, and click a result to open their full player modal.",
        "link": "/players"
    },
    {
        "date": "2026-05-13",
        "tag": "feature",
        "text": "Player Scoring: Player modals now show PPG and season total points side by side (e.g. 24.5 | 416.5) with positional ranks for both — PPG · RB1 | TOTAL · RB2. Compare player view shows the same scoring breakdown for both players.",
        "link": "/players"
    },
    {
        "date": "2026-05-13",
        "tag": "feature",
        "text": "Player Rankings: Sort by PPG or Total Points to quickly find the highest scorers. The sort column shows the value and positional rank so you can see where each player stands.",
        "link": "/players"
    },
    {
        "date": "2026-05-13",
        "tag": "new",
        "text": "Playoff Odds: See your chances of making the playoffs, earning a first-round bye, and winning the championship based on your roster and schedule on the Teams page.",
        "link": "/teams"
    },
    {
        "date": "2026-05-11",
        "tag": "new",
        "text": "Start/Sit Advisor: Weekly lineup recommendations on the Waivers page. Shows your starters, FLEX slot picks, and bench players ranked by projected points with opponent matchup adjustments.",
        "link": "/waivers"
    },
    {
        "date": "2026-05-11",
        "tag": "new",
        "text": "Waiver Wire: Ranked free agent targets on the Waivers page. Filter by position and see pickup signals.",
        "link": "/waivers"
    },
    {
        "date": "2026-05-11",
        "tag": "feature",
        "text": "Trade Database: Search with multiple players on each side of a trade to find exact multi-player packages from real dynasty leagues.",
        "link": "/trade-database"
    },
    {
        "date": "2026-05-02",
        "tag": "new",
        "text": "Draft Grades: New draft tab on the Teams page grades every rookie draft pick - ADP value, positional need, and best player available. View by team or by round. ADP sourced from real drafts in similar leagues when available.",
        "link": "/teams"
    },
    {
        "date": "2026-04-29",
        "tag": "new",
        "text": "League Superlatives: 10 all-time awards on the Awards page - Barely Breathing (most wins by <5 pts), Consistency King, Main Character, Bench Warmer MVP, Waiver Wire Demon, Playoff Riser, plus The Bridesmaid, Most Dominant, The Punching Bag, and Boom or Bust.",
        "link": "/awards"
    },
    {
        "date": "2026-04-25",
        "tag": "new",
        "text": "Historical Comparables: Rookie prospects now show historical player comps based on position, prospect score, and athletic profile. See how similar prospects performed in the NFL.",
        "link": "/prospects"
    },
    {
        "date": "2026-04-23",
        "tag": "new",
        "text": "Trade Outcome: New historical value analysis for past trades. Shows what players were worth at trade date vs current value to evaluate trade performance over time",
        "link": "/activity"
    },
    {
        "date": "2026-04-23",
        "tag": "new",
        "text": "Trade Database: Browse real dynasty trades from thousands of leagues, search by player name, filter by league type, and see actual trade packages.",
        "link": "/trade-database"
    },
    {
        "date": "2026-04-23",
        "tag": "new",
        "text": "Trade Intelligence: Advanced market analytics showing real trade frequency, market values, and momentum trends.",
        "link": "/trade-intel"
    },
    {
        "date": "2026-04-23",
        "tag": "new",
        "text": "Roster Grades: AI-powered roster evaluation with letter grades based on positional strength, age curves, and championship probability.",
        "link": "/teams"
    },
    {
        "date": "2026-04-23",
        "tag": "new",
        "text": "Roster Archetypes: Identify your team's competitive window: Win-Now Window, Rising Contender, 2-3 Year Window, Full Rebuild, or Retooling.",
        "link": "/teams"
    },
    {
        "date": "2026-04-18",
        "tag": "new",
        "text": "Live NFL News: Player modals now show real ESPN headlines for each player. Activity page includes a live NFL news feed with the latest league headlines.",
        "link": "/activity"
    },
    {
        "date": "2026-04-18",
        "tag": "new",
        "text": "Trade Targets: Trade Calculator sidebar shows position-gap-based trade targets for your roster-identifies what you need most and which players to target by owner.",
        "link": "/trade"
    },
    {
        "date": "2026-04-18",
        "tag": "feature",
        "text": "Trade Calculator: Auto-selects your team on load so you see your roster and trade targets immediately without manual selection.",
        "link": "/trade"
    },
    {
        "date": "2026-04-18",
        "tag": "new",
        "text": "Roster Intel: Team analytics now includes a Roster Intel tab with per-player signals - Core, Sell High, Buy Window, Breakout Hold, and Cut recommendations based on value trends and age curves.",
        "link": "/teams"
    },
    {
        "date": "2026-04-18",
        "tag": "new",
        "text": "Waiver Wire Targets: Offseason waiver recommendations ranked by pickup score - combines value, 7-day trend, breakout signals, and age-prime windows to surface the best available adds.",
        "link": "/trade"
    },
    {
        "date": "2026-04-18",
        "tag": "feature",
        "text": "Player Rankings: 7-day rank movement indicators (▲/▼) on every player showing how many spots they've moved in the last week.",
        "link": "/players"
    },
    {
        "date": "2026-04-17",
        "tag": "new",
        "text": "Team Analytics: Beat the Market portfolio analytics showing 30-day value changes compared to league average, plus new graphs and charts showing portfolio value trends. See which teams are gaining or losing value with key mover breakdowns.",
        "link": "/teams"
    },
    {
        "date": "2026-04-15",
        "tag": "feature",
        "text": "Player Comparison: Compare any two players with position-specific stats and metrics. See how players stack up with relevant comparisons for their positions.",
        "link": "/dashboard"
    },
    {
        "date": "2026-04-14",
        "tag": "feature",
        "text": "Stats tab: Awards (all-time league records and championship history), Graphs (career aggregate view + per-season breakdown), and History are now combined into one dropdown for easier navigation. History data preloads in the background after login so pages open instantly.",
        "link": None,
    },
    {
        "date": "2026-04-09",
        "tag": "new",
        "text": "Rookie Rankings: Full prospect evaluation system: production, athleticism, draft capital, and dynasty value for the active draft class",
        "link": "/prospects"
    },
    {
        "date": "2026-04-09",
        "tag": "new",
        "text": "Player Rankings: New dedicated page with searchable, filterable player rankings: filter by position (multi-select), league format, team count, and sort by rank, value, age, or positional rank",
        "link": "/players"
    },
    {
        "date": "2026-04-09",
        "tag": "new",
        "text": "Breakouts Page: Dedicated dashboard for breakout candidates with opportunity projections and scoring",
        "link": "/breakouts"
    },
    {
        "date": "2026-04-01",
        "tag": "feature",
        "text": "Advanced Metrics: View role scores, snap share, and efficiency stats in player modals",
        "link": "/dashboard"
    },
    {
        "date": "2026-04-01",
        "tag": "feature",
        "text": "Dark Mode: Toggle between light and dark themes with your preference saved automatically",
        "link": "/dashboard"
    },
    {
        "date": "2026-04-01",
        "tag": "feature",
        "text": "Value Movers: Filter by 7-day, 14-day, or 30-day periods to track player value changes",
        "link": "/trade"
    },
    {
        "date": "2026-03-31",
        "tag": "feature",
        "text": "Player & team modals: click any player or team to view detailed stats and roster info.",
        "link": "/dashboard"
    },
    {
        "date": "2026-03-30",
        "tag": "new",
        "text": "Offseason Breakouts: Identifies dynasty breakout candidates before the season starts based on roster changes",
        "link": "/trade"
    },
    {
        "date": "2026-03-30",
        "tag": "feature",
        "text": "Trade Calculator: Value changes now show +/- indicators on player chips",
        "link": "/trade"
    },
    {
        "date": "2026-03-30",
        "tag": "feature",
        "text": "Trade Calculator: Shareable trade links - copy and send trades to league mates instantly",
        "link": "/trade"
    },
    {
        "date": "2026-03-30",
        "tag": "feature",
        "text": "Trade Calculator: Rookie and breakout badges identify emerging talent",
        "link": "/trade"
    },
    {
        "date": "2026-03-30",
        "tag": "feature",
        "text": "Top Movers: See when player values were last updated with freshness indicators",
        "link": "/trade"
    },
    {
        "date": "2026-03-27",
        "tag": "new",
        "text": "Team Reports: AI-powered Front Office Briefings analyze your roster, trade opportunities, and league standings",
        "link": "/dashboard"
    },
    {
        "date": "2026-03-26",
        "tag": "new",
        "text": "Trade Calculator: AI analysis now personalizes to your roster and team direction",
        "link": "/trade"
    },
    {
        "date": "2026-03-25",
        "tag": "new",
        "text": "History Page: Generate AI season recaps personalized to your team's storyline",
        "link": "/history"
    },
    {
        "date": "2026-03-23",
        "tag": "feature",
        "text": "Trade Calculator: Analysis auto-updates as you add or remove players",
        "link": "/trade"
    },
    {
        "date": "2026-03-22",
        "tag": "feature",
        "text": "Trade Calculator: Counter suggestions now include specific players and picks",
        "link": "/trade"
    }
]
