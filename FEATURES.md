# BR Fantasy — Comprehensive Feature List

A full breakdown of every feature on the site, organized by the main navigation.

---

## Platform & Account

- **Multi-platform support** — Connect dynasty leagues from **Sleeper** (username sign-in) and **ESPN** (league ID validation); **Yahoo** OAuth import in progress.
- **Username sign-in / Identify** — Log in with just a Sleeper username; the app finds all your leagues automatically.
- **Google sign-in** — Sign in with a Google account for cross-device, account-scoped state (watchlist sync, one-time "since last visit" digests).
- **Keeper & redraft leagues** — Automatic detection of keeper-eligible and redraft leagues; keeper-specific tools and nav appear only where relevant.
- **My Leagues (Portfolio)** — Cross-league hub that lists every league you're in with at-a-glance value and standings.
- **League switcher** — Jump between your leagues from anywhere via the nav dropdown.
- **Multi-season support** — View any league across multiple seasons.
- **PRO / Premium tiers** — Subscription-gated features (Stripe checkout, billing portal, league-shared or user-based premium access) with a paywall on PRO-only tools.
- **Dark mode** — Light/dark theme toggle with your preference saved automatically.
- **Changelog** — In-app feed of every new feature and update.

---

## Dashboard

- **Front Office Report** — AI-generated report analyzing your roster, trade opportunities, and standings, personalized to your team.
- **Since Your Last Visit** — Personalized digest of league activity (trades, waivers) plus your roster's value moves and new injuries since you were last on. Google-account visits consume the digest server-side, so it's a true one-time, cross-device notification; signed-out visitors get a local-browser fallback.
- **Offseason Hub** — Offseason team snapshot, draft countdown, and Draft Capital Index.
- **Rookie Draft Assistant (preview)** — Surfaced on the dashboard during draft season.
- **Waiver Wire Targets** — Top available adds ranked for your roster.
- **League leader callouts** — Quick stat highlights for the league.
- **League Bulletins** — Surfaces your Sleeper league's bulletin-board messages in-app.

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
  - **Scout Report** — Opponent breakdown for your current-week matchup (regular season, signed-in).
  - **Top Scorers** — Highest scorers for the week.
  - **Power Rankings** — Weekly power ranking of teams.
  - **Strength of Schedule (SOS)** — Schedule difficulty breakdown.
- **Redzone (live)** — Live red-zone / scoring tracker with league-wide and your-team scopes. Works on every platform (Sleeper, ESPN, Yahoo, MFL) — providers canonicalize player ids to a common feed, and live stat lines come from Tank01 boxscores. Cross-league "My Leagues" uses the signed-in account portfolio on every platform (and still walks Sleeper leagues for a Sleeper-only session).
- **Streaming Options** — Matchup-based D/ST and K streaming targets from free agents, ranked by opponent Vegas implied totals; gated to positions your league actually starts.
- **Weekly Recap** — AI-written recap of the week with a shareable OG share image.

---

## League

- **Standings** — League standings and records.
- **Teams** — Deep team analytics with tabs:
  - **Roster Grades** — AI letter grades from positional strength, age curves, and championship probability.
  - **Roster Intel** — Per-player signals: Core, Sell High, Buy Window, Breakout Hold, Monitor, Cut.
  - **Roster Archetypes** — Competitive window: Win-Now, Rising Contender, 2-3 Year Window, Full Rebuild, Retooling.
  - **Playoff Odds** — Chances of making playoffs, earning a bye, and winning the title.
  - **Playoff Scenarios** — Deterministic end-of-season clinch/elimination picture: who has clinched, who is eliminated, who controls their own destiny, magic numbers, and "win-and-you're-in" swings (exact inside the final-weeks window; falls back to odds earlier). Division leagues seed division winners first, then wild cards.
  - **Power Rankings** — Team power rankings.
  - **Beat the Market (Portfolio)** — 30-day value trends vs. league average with key-mover breakdowns and charts.
  - **Draft Grades** — Grades every rookie draft pick (ADP value, positional need, best player available); view by team or round.
- **Activity** — League transaction feed plus a live NFL news feed of the latest headlines.
- **League Health** — Multi-season league health view with trend tracking that only compares completed seasons (no partial-season skew).
- **Commissioner** — Dedicated commissioner view for league-level oversight.

---

## Players

- **Player Rankings** — Searchable, filterable rankings by position (multi-select), league format, and team count; sort by rank, value, age, PPG, or total points, with positional ranks and 7-day rank-movement indicators (▲/▼).
- **Player Search** — Nav-bar search (magnifying glass / Ctrl+K) to open any player's modal.
- **Watchlist** — Star any player to a personal watchlist (local-first, synced to your account when signed in so it follows you across devices). Watched players surface **value-move and injury alerts** — flagged when a player moves past the value threshold over 7 days or picks up a real injury designation.
- **Player Modals** — Detailed player view: PPG and season total with positional ranks, advanced metrics (snap share, role score, efficiency), career/per-season game logs, value history, live ESPN headlines, and a **Trades** tab that toggles between **This League** (every season, real counterparties, picks resolved to drafted players) and the **Trade Database** (same free cross-league comps, also with pick→player resolution when drafts are complete).
- **Player Comparison** — Compare any two players with position-specific stats and metrics.
- **Prospect Rankings** — Full rookie evaluation: production, athleticism, draft capital, and dynasty value for the active class, plus historical player comps.
- **Draft Assistant** — Draft Board that analyzes positional needs vs. the league, recommends 1–2 prospects per pick, shows ADP and grade per row, and tracks who you've drafted this session.
  - **Mock Draft simulator** — Run a full mock draft against simulated opponents from the draft room.
  - **Cheat Sheet** — Sortable, printable draft cheat sheet with an embeddable version for sharing.
  - **Draft History** — Review completed drafts.
- **Keeper Assistant** — For keeper leagues: auto-detects each player's draft-round keeper cost from Sleeper, Yahoo, and ESPN drafts, then picks the best keepers under your league's keeper limit and cost rules, with a full sortable table and live re-calc as you tweak the limit.
- **Breakout Engine (PRO)** — Breakout candidates with opportunity projections, vacated-target totals, historical peer comps from real breakout seasons, and confidence-adjusted projected PPG ranges. Also includes offseason breakout candidates.
- **Waivers & Start/Sit** — Ranked free-agent targets with pickup signals (filter by position), plus a weekly Start/Sit Advisor showing starters, FLEX picks, and bench ranked by start score. Kicker and D/ST are included when the league starts them.
- **Schedule Assistant** — Matchup difficulty for every rostered player across any chosen week range, with add/remove players.

---

## Stats

- **Awards** — All-time league records, championship history, and 10 league superlatives (Barely Breathing, Consistency King, Main Character, Bench Warmer MVP, Waiver Wire Demon, Playoff Riser, The Bridesmaid, Most Dominant, The Punching Bag, Boom or Bust).
- **Graphs** — Career aggregate view plus per-season breakdowns of league value and performance trends.
- **History** — Season-by-season standings and summaries, plus AI season recaps personalized to your team's storyline (preloaded in the background for instant opening).

---

## Content, SEO & Sharing

- **Public landing pages** — Unauthenticated, SEO-focused surfaces that work without a league: dynasty rankings (overall and per-position QB/RB/WR/TE), a **Dynasty Trade Value Chart**, **Player Compare**, **Prospects**, **Breakouts**, **Top Movers**, and per-player pages (`/player/<slug>` and `/player/<slug>/trade-value`).
- **Guides & Glossary** — Long-form dynasty guides (e.g. trade-value strategy) and a fantasy-term glossary.
- **Share Cards** — Shareable team/roster cards with generated OG images for posting to league chats and social; trades are shareable via `/t/<id>` and `/trade-card/<id>` links with their own OG images.
- **Sitemap / robots** — Generated `sitemap.xml` and `robots.txt` covering guides and top player pages.

---

## Under the Hood

- **Live value engine** — Player values with 7/14/30-day movers, freshness indicators, and value-history tracking.
- **Monte Carlo simulation** — Powers playoff odds and the trade Playoff Impact card.
- **Real-trade crawler** — Aggregates dynasty trades across thousands of leagues for the database and intel tools.
- **NFL state / news integration** — Live NFL week state, player news, and injury data.
- **Responsive design** — Container-query-driven layouts that adapt cleanly from desktop to mobile, with a mobile tab-bar dock.
- **PWA & offline** — Installable progressive web app (service worker, manifest, offline page) plus push notifications for trades, breakouts, waivers, and scores.
- **Weekly email digest** — Once-a-week recap emailed to signed-in users: your record and league rank, your roster's value risers/fallers, and the biggest leaguewide movers, linking back to your dashboard. De-duped per account per week, with one-click signed unsubscribe. Sent via the `/api/cron/notifications` hook (`type=weekly`).
- **Browser extension** — Companion extension for reading league/player context on Sleeper.
- **Trending surfaces** — Trending adds, risers/fallers, and value-movers boards driven by the live value engine.
- **Static / informational pages** — About, Pricing, FAQ, Contact, Support, Privacy, Terms.
