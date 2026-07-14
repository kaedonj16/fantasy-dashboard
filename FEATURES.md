# BR Fantasy — Comprehensive Feature List

A full breakdown of every feature on the site, organized by the main navigation.

---

## Platform & Account

- **Multi-platform support** — Connect dynasty leagues from **Sleeper** (username sign-in) and **ESPN** (league ID validation).
- **Username sign-in / Identify** — Log in with just a Sleeper username; the app finds all your leagues automatically.
- **My Leagues (Portfolio)** — Cross-league hub that lists every league you're in with at-a-glance value and standings.
- **League switcher** — Jump between your leagues from anywhere via the nav dropdown.
- **Multi-season support** — View any league across multiple seasons.
- **PRO / Premium tiers** — Subscription-gated features (Stripe checkout, billing portal, league-shared or user-based premium access) with a paywall on PRO-only tools.
- **Dark mode** — Light/dark theme toggle with your preference saved automatically.
- **Changelog** — In-app feed of every new feature and update.

---

## Dashboard

- **Front Office Report** — AI-generated report analyzing your roster, trade opportunities, and standings, personalized to your team.
- **Offseason Hub** — Offseason team snapshot, draft countdown, and Draft Capital Index.
- **Rookie Draft Assistant (preview)** — Surfaced on the dashboard during draft season.
- **Waiver Wire Targets** — Top available adds ranked for your roster.
- **League leader callouts** — Quick stat highlights for the league.

---

## Trades

- **Trade Calculator** — Compare both sides of a deal using BR values, balance, and roster-building context.
  - Auto-selects your team on load so you see your roster and targets immediately.
  - Value +/- indicators and rookie/breakout badges on player chips.
  - Shareable trade links to send deals to league mates.
  - AI trade analysis personalized to your roster and team direction; auto-updates as you add/remove players.
  - Counter-suggestions with specific players and picks.
  - Format controls (team count, PPR/scoring, 1QB/Superflex).
- **Playoff Impact** — Live Monte Carlo simulation of how a trade shifts your **playoff odds, projected wins, projected PPG, top-3 draft-pick odds, roster age, and prime years left**, with a plain-language verdict (Win-Now Move, Building Move, Balanced, etc.) and metric-explanation tooltip.
- **Trade Suggestions (PRO)** — Archetype-driven packages for all four team profiles (**Contending, Rebuilding, Consolidate, Distribute**); each runs a full post-trade playoff sim so the Win% / playoff-odds shifts reflect the real net effect. Consolidate only surfaces genuine trade-up packages.
- **Trade Targets** — Position-gap-based targets identifying what your roster needs most and which players to target by owner.
- **Trade Intel (PRO)** — Market analytics with real trade frequency, market values, and momentum trends; shows how people actually acquire a given player, with packages loadable into the calculator in one click.
- **Trade Database** — Browse real dynasty trades from thousands of leagues; search by single or multiple players per side, filter by league type, and see actual packages.
- **Trade Outcome** — Historical value analysis of past trades (player value at trade date vs. current).

---

## Weekly

- **Matchups / Weekly Hub** — Weekly matchup view with multiple tabs:
  - **Optimal Lineup** — Best possible lineup vs. what was started.
  - **Top Scorers** — Highest scorers for the week.
  - **Power Rankings** — Weekly power ranking of teams.
  - **Strength of Schedule (SOS)** — Schedule difficulty breakdown.
- **Weekly Recap** — AI-written recap of the week with a shareable OG share image.

---

## League

- **Standings** — League standings and records.
- **Teams** — Deep team analytics with tabs:
  - **Roster Grades** — AI letter grades from positional strength, age curves, and championship probability.
  - **Roster Intel** — Per-player signals: Core, Sell High, Buy Window, Breakout Hold, Monitor, Cut.
  - **Roster Archetypes** — Competitive window: Win-Now, Rising Contender, 2-3 Year Window, Full Rebuild, Retooling.
  - **Playoff Odds** — Chances of making playoffs, earning a bye, and winning the title.
  - **Power Rankings** — Team power rankings.
  - **Beat the Market (Portfolio)** — 30-day value trends vs. league average with key-mover breakdowns and charts.
  - **Draft Grades** — Grades every rookie draft pick (ADP value, positional need, best player available); view by team or round.
- **Activity** — League transaction feed plus a live NFL news feed of the latest headlines.
- **League Health (Commissioner)** — Multi-season league health view with trend tracking that only compares completed seasons (no partial-season skew).

---

## Players

- **Player Rankings** — Searchable, filterable rankings by position (multi-select), league format, and team count; sort by rank, value, age, PPG, or total points, with positional ranks and 7-day rank-movement indicators (▲/▼).
- **Player Search** — Nav-bar search (magnifying glass / Ctrl+K) to open any player's modal.
- **Player Modals** — Detailed player view: PPG and season total with positional ranks, advanced metrics (snap share, role score, efficiency), career/per-season game logs, value history, and live ESPN headlines.
- **Player Comparison** — Compare any two players with position-specific stats and metrics.
- **Prospect Rankings** — Full rookie evaluation: production, athleticism, draft capital, and dynasty value for the active class, plus historical player comps.
- **Draft Assistant** — Draft Board that analyzes positional needs vs. the league, recommends 1–2 prospects per pick, shows ADP and grade per row, and tracks who you've drafted this session.
- **Breakout Engine (PRO)** — Breakout candidates with opportunity projections, vacated-target totals, historical peer comps from real breakout seasons, and confidence-adjusted projected PPG ranges. Also includes offseason breakout candidates.
- **Waivers & Start/Sit** — Ranked free-agent targets with pickup signals (filter by position), plus a weekly Start/Sit Advisor showing starters, FLEX picks, and bench ranked by projected points with matchup adjustments.
- **Schedule Assistant** — Matchup difficulty for every rostered player across any chosen week range, with add/remove players.

---

## Stats

- **Awards** — All-time league records, championship history, and 10 league superlatives (Barely Breathing, Consistency King, Main Character, Bench Warmer MVP, Waiver Wire Demon, Playoff Riser, The Bridesmaid, Most Dominant, The Punching Bag, Boom or Bust).
- **Graphs** — Career aggregate view plus per-season breakdowns of league value and performance trends.
- **History** — Season-by-season standings and summaries, plus AI season recaps personalized to your team's storyline (preloaded in the background for instant opening).

---

## Under the Hood

- **Live value engine** — Player values with 7/14/30-day movers, freshness indicators, and value-history tracking.
- **Monte Carlo simulation** — Powers playoff odds and the trade Playoff Impact card.
- **Real-trade crawler** — Aggregates dynasty trades across thousands of leagues for the database and intel tools.
- **NFL state / news integration** — Live NFL week state, player news, and injury data.
- **Responsive design** — Container-query-driven layouts that adapt cleanly from desktop to mobile.
- **Static / informational pages** — About, Pricing, FAQ, Contact, Support, Privacy, Terms.
