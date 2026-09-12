# BR Fantasy — Comprehensive Feature List

A full breakdown of every feature on the site, organized by the main navigation.

---

## Platform & Account

- **Multi-platform support**: Connect leagues from **Sleeper** (username sign-in), **ESPN** (league ID; public or private cookie connect), **Yahoo** (OAuth), **MFL** (league ID, public or private auth), and **Fleaflicker** (league ID, public or private email/token auth). ESPN football does not expose future draft picks, so draft capital stays empty rather than invented. ESPN leagues are treated as redraft.
- **Username sign-in / Identify** — Log in with just a Sleeper username; the app finds all your leagues automatically.
- **Google sign-in** — Sign in with a Google account for cross-device, account-scoped state (watchlist sync, one-time "since last visit" digests).
- **Keeper & redraft leagues**: Automatic detection of keeper-eligible and redraft leagues; keeper-specific tools and nav appear only where relevant. **Auction** leagues are detected from provider draft settings (Sleeper/ESPN/MFL signals); Draft Room and Keeper show an honest provisional banner. Snake-round draft grades stay off for auction. Auction nomination guidance (suggested $ from BR value vs remaining budget/slots) appears in Draft Room — not a clearing-price model. **Best Ball** leagues get a Season Hub badge, nav labeled Waivers (no Start/Sit), and Start/Sit hidden on the waivers page.
- **My Leagues (Portfolio)** — Cross-league hub that lists every league you're in with at-a-glance value and standings. **PRO** unlocks the cross-league **This week's moves** digest (lineup, injury, and waiver adds ranked across linked leagues). Waiver cards only surface startable or need-filling names — leftover RB35 / streamer-QB wire fodder is omitted.
- **League switcher** — Jump between your leagues from anywhere via the nav dropdown.
- **Multi-season support** — View any league across multiple seasons.
- **PRO / Premium tiers** — Subscription-gated features (Stripe checkout, billing portal) with a paywall on PRO-only tools. Plans: **One League** ($5/year, buyer-only for one selected league), **Personal** ($10/year, all your leagues), **League** ($15/year, shared with every manager), and **League + Personal** ($20/year combo).
- **Dark mode** — Light/dark theme toggle with your preference saved automatically.
- **Changelog** — In-app feed of every new feature and update.

---

## Dashboard

- **Front Office Report (PRO)** — AI-generated report analyzing your roster, trade opportunities, and standings, personalized to your team. Both free and PRO users see a **Generate Report** control on the in-season and offseason hubs; free users hit the paywall, PRO users generate on click. The memo is never auto-served on page load.
- **Since Your Last Visit** — Personalized digest of league activity (trades, waivers) plus your roster's value moves and new injuries since you were last on. Google-account visits consume the digest server-side, so it's a true one-time, cross-device notification; signed-out visitors get a local-browser fallback.
- **Offseason Hub** — Offseason team snapshot, draft countdown, and Draft Capital Index.
- **Waiver Wire Targets** — Top available adds ranked for your roster.
- **League leader callouts** — Quick stat highlights for the league.
- **League Bulletins** — Turned off on every platform (including Sleeper). The API always returns unavailable; the dashboard no longer renders a bulletins card.

---

## Trades

- **Trade Calculator** — Compare both sides of a deal using BR values, balance, and roster-building context.
  - Auto-selects your team on load so you see your roster and targets immediately.
  - Value +/- indicators and rookie/breakout badges on player chips.
  - Shareable trade links to send deals to league mates.
  - **AI trade analysis (PRO)** personalized to your roster and team direction; auto-updates as you add/remove players.
  - **Counter-suggestions (PRO)** with specific players and picks.
  - Format controls (team count, PPR/scoring, 1QB/Superflex).
- **Playoff Impact (PRO)** — Live Monte Carlo simulation of how a trade shifts your **playoff odds, projected wins, and projected PPG**, with a plain-language verdict (Win-Now Move, Building Move, Balanced, etc.) and metric-explanation tooltip. Gated in the UI and on the API. Dynasty leagues also get a **Future Outlook** block (top-3 draft-pick odds, roster age, prime years left); those fields are stripped for redraft.
- **Trade Suggestions (PRO)** — Archetype-driven packages for all four team profiles (**Contending, Rebuilding, Consolidate, Distribute**); each runs a full post-trade playoff sim so the Win% / playoff-odds shifts reflect the real net effect. Consolidate only surfaces genuine trade-up packages.
- **Trade Targets (PRO)** — Roster-fit targets for your gaps: players that upgrade the hole at a price you can pay, from teams that need your surplus, mixed across positions — not the top four names at a weak spot.
- **Trade Intel (PRO)** — Market analytics with real trade frequency, market values, and momentum trends; shows how people actually acquire a given player, with packages loadable into the calculator in one click. Shown in nav on every platform (including Sleeper and Fleaflicker), with a note that comps are sourced from Sleeper dynasty trades.
- **Trade Database** — Browse real dynasty trades from thousands of leagues; search by single or multiple players per side, filter by league type, and see actual packages.
- **Trade Outcome** — Historical value analysis of past trades (player value at trade date vs. current).

---

## Weekly

- **Matchups / Weekly Hub** — Weekly matchup view with tabs:
  - **Matchups** — Current-week matchup cards (default).
  - **Scorers** — Highest scorers for the week.
  - **Scout** — Opponent breakdown for your current-week matchup (regular season, signed-in).
  - **Lineup** — Best possible lineup vs. what was started.
- **Redzone (live)** — Live red-zone / scoring tracker with league-wide and your-team scopes. In the Weekly nav on every platform (Sleeper, ESPN, Yahoo, MFL, Fleaflicker) — providers canonicalize player ids to a common feed, and live stat lines come from Tank01 boxscores. Cross-league "My Leagues" uses the signed-in account portfolio on every platform (and still walks Sleeper leagues for a Sleeper-only session).
- **Weekly Recap** — AI-written recap of the week with a shareable OG share image. The AI storyline is PRO; free users see an upgrade teaser. Sample/preview weeks are labeled as sample data, not your league’s results.

---

## League

- **Standings** — League standings and records, with Detailed Stats and Value Share tabs. The Standings power card also hosts **Power Rankings**, optional **Playoff Picture** (when the bracket is available), and **Playoff Odds**. Late in the season, Playoff Odds gains an **Outlook** column from deterministic clinch/elimination math (who has clinched, who is eliminated, who controls their own destiny, magic numbers, and "win-and-you're-in" swings — exact inside the final-weeks window; falls back to odds earlier). Division leagues seed division winners first, then wild cards.
- **Teams** — Deep team analytics. The team grid shows **roster letter grades** and **competitive-window / archetype** labels (sortable on the grid). Sidebar / mobile tabs:
  - **Value (Beat the Market)** — 30-day value trends vs. league average with key-mover breakdowns and charts.
  - **Roster Intel** — Per-player signals: Core, Sell High, Buy Window, Breakout Hold, Monitor, Cut.
  - **Schedule (SOS)** — Matchup difficulty for rostered players (shown in-season / preseason, not pure offseason).
- **Activity** — League transaction feed plus a live NFL news feed of the latest headlines.
- **League Health** — Multi-season league health view with trend tracking that only compares completed seasons (no partial-season skew). The same page includes commissioner oversight extras such as a copyable League PRO invite link for teammates (there is no separate Commissioner nav item).

---

## Players

- **Player Rankings** — Searchable, filterable rankings by position (multi-select), league format, and team count; sort by rank, value, age, PPG, or total points, with positional ranks and 7-day rank-movement indicators (▲/▼).
- **Player Search** — Nav-bar search (magnifying glass / Ctrl+K) to open any player's modal.
- **Watchlist** — Star any player to a personal watchlist (local-first, synced to your account when signed in so it follows you across devices). Watched players surface **value-move and injury alerts** — flagged when a player moves past the value threshold over 7 days or picks up a real injury designation.
- **Player Modals** — Detailed player view: PPG and season total with positional ranks, advanced metrics (snap share, role score, efficiency), career/per-season game logs, value history, live ESPN headlines, and a **Team** tab with position-aware role/competition, compact schedule preview, season results and team environment. Teammate links retain the inspected season and offer an in-modal return path. The **Trades** tab toggles between **This League** (every season, real counterparties, picks resolved to drafted players) and the **Trade Database** (same free cross-league comps, also with pick→player resolution when drafts are complete).
- **Player Comparison** — Compare any two players with position-specific stats and metrics.
- **Advanced Metrics** — League-aware advanced-metrics leaderboard (snap share, role, efficiency, and related signals). Free on pricing; not paywalled.
- **Prospect Rankings** — Full rookie evaluation: production, athleticism, draft capital, and dynasty value for the active class, plus historical player comps.
- **Draft Room**: Live and mock draft board with best-available rankings, pick scoring, and post-draft grades. Run a mock for any format or connect to a live **Sleeper, ESPN, or Yahoo** draft (MFL live sync is unsupported). Custom Cheat Sheet board (pin, mute, reorder) follows PRO users into the Draft Room. **Deep Dive (PRO)** replays Decision Score against the historical remaining pool. Supports rookie, startup, and redraft leagues.
  - **Mock Draft simulator**: Run a full mock draft against simulated opponents from the draft room.
  - **Cheat Sheet**: Sortable, printable draft cheat sheet with opt-in **Connect live draft** sync for the league's platform (free; ESPN uses sync=1), free CSV download, and an embeddable in-draft overlay that stays crossed-off as picks land. Custom board edits stay PRO. **Trend Scout (PRO)** surfaces historical ranking/ADP trends on the sheet.
  - **Draft History** — Review completed drafts and per-pick grades (by team or round). Draft grades live here / in Draft Room — not as a Teams sidebar tab.
- **Keeper Assistant** — For keeper leagues: auto-detects each player's draft-round keeper cost from Sleeper, Yahoo, and ESPN drafts (and years-kept on Sleeper season chains / ESPN keeper flags), then picks the best keepers under your league's keeper limit and cost rules, with a full sortable table and live re-calc as you tweak the limit. Auction/FAAB keeper costs are not auto-detected — set the drafted round by hand. MFL/Fleaflicker costs stay manual.
- **Breakout Engine (PRO)** — Breakout candidates with opportunity projections, vacated-target totals, historical peer comps from real breakout seasons, and confidence-adjusted projected PPG ranges. Also includes offseason breakout candidates. If roster-change data has not been loaded for the season, the board stays empty instead of ranking players on readiness alone.
- **Waivers & Start/Sit** — Ranked free-agent targets with pickup signals (filter by position), plus a weekly Start/Sit Advisor showing starters, FLEX picks, and bench ranked by start score. Kicker and D/ST are included when the league starts them. FAAB leagues get **low · target · stretch** bid bands with a short rationale, drop suggestions when the roster is full, and approximate schedule-urgency notes. Hourly waiver pushes deep-link into Waivers with the week's top available add. **Streaming this week** (matchup-based D/ST and K targets from free agents, ranked by opponent Vegas implied totals; gated to positions your league actually starts) lives on the Waivers page, not as a Weekly Hub tab.
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
- **PWA & offline** — Installable progressive web app (service worker, manifest, offline page) plus push notifications for trades, breakouts, waivers, and scores. **Lineup-lock** pushes (and the matching in-app toast) include concrete start/sit swaps such as “Sit A for B (+X.X proj)” when projections show points on the bench, including when a starter is injured or on bye. Rate limits use Redis when `REDIS_URL` is set so they hold across web workers; otherwise they are per-process.
- **Weekly email digest** — Personalized Tuesday recap (Render cron `weekly-email`, 13:00 UTC) covering **every connected league**. One-league accounts still get the full recap; two or more leagues get a snapshot table (record + one focus line each) plus lineup/injury bullets, with a CTA to My Leagues (`/portfolio`). Redraft focus lines lead with matchup / start-sit; dynasty lines lead with risers and top assets. Delivered via Brevo (`htmlContent`) with SMTP fallback. De-duped per account per ISO week, with a signed unsubscribe link that opts out of the weekly digest only. See `docs/weekly-email.md`. Hourly push checks (lineup lock, close games, drops, injuries) run from Render cron `hourly-notifications`.
- **Browser extension**: Companion extension for ESPN/Yahoo live-draft relay, one-click ESPN private-league connect, and a read-only Draft Assistant overlay on **Sleeper, Yahoo, and ESPN** draft rooms. See `extension/README.md` for the parity checklist (phone drafts stay manual track; production zip via `pack_extension.py`).
- **Trending surfaces** — Trending adds, risers/fallers, and value-movers boards driven by the live value engine.
- **Static / informational pages** — About, Pricing, FAQ, Contact, Support, Privacy, Terms.
