"""
Changelog entries for the fantasy dashboard.
Each entry represents a user-facing change.
"""

CHANGELOG = [
    {
        "date": "2026-10-27",
        "tag": "update",
        "text": "Weekly Hub: the live matchup views now refresh with far less repeated work behind them. The drive bars read a small game-state feed instead of the full live payload, the matchup board and the player popup share one live-data cache instead of fetching the same feed twice, a refresh whose numbers have not moved is skipped entirely, and the schedule and rankings lookups behind every matchup slide are built once per week instead of once per slide. Nothing on the page changes: same numbers, same panels, just less downloading and recomputing.",
        "link": "/"
    },
    {
        "date": "2026-10-27",
        "tag": "fix",
        "text": "Weekly menu: removed a duplicate red dot. On Tuesdays the recap-ready banner added its own dot to the Weekly menu on top of the Weekly Recap unread dot, so two dots showed at once. The banner still appears on Tuesdays, and the single recap dot remains and clears when you open the recap.",
        "link": "/recap"
    },
    {
        "date": "2026-10-27",
        "tag": "fix",
        "text": "My Leagues: the phone section buttons (Strength, Rooting Guide, Byes, Holdings) now land each section right below the top bars instead of scrolling its header underneath them. The Playoff odds stat is off the league cards: it never loaded, so the stats row is Record, Standing and Streak, and the sort control now offers Standings, which orders your leagues by their place in the table.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-27",
        "tag": "fix",
        "text": "Notifications on your phone: the panel now fits the screen. It stops above the bottom nav instead of sliding underneath it, the list scrolls inside the panel while the header stays in place, and the page behind it dims while it is open. The red alert dot on the More tab also sits on the More icon again instead of hanging under the label.",
        "link": "/"
    },
    {
        "date": "2026-10-27",
        "tag": "fix",
        "text": "Weekly Recap: Week at a Glance now shows the manager username under the High Scorer and Low Scorer team names. It was repeating the team name there instead, so the High Scorer card read Fire Riley with @Fire Riley underneath it.",
        "link": "/recap"
    },
    {
        "date": "2026-10-27",
        "tag": "update",
        "text": "Weekly Hub: the matchup page is rebuilt around your game. Week Leaders now run across the top, your matchup carousel sits on the left and stays in view while the other column scrolls, and Scorers, Scout and Lineup live in one panel on the right. Scout is now a gameplan: the verdict with your win probability, the calls that decide the matchup, the positional edge, their top threats and their soft spots. Lineup opens on This Week, your current starters against the projected-optimal lineup, the swaps that gain you points and what they do to your win probability, with the completed-weeks Lineup Lab under History.",
        "link": "/"
    },
    {
        "date": "2026-10-27",
        "tag": "update",
        "text": "League Stats: the chart grid now fits two columns only when each chart has room and drops to one column on narrower windows, cards in a row stretch to the same height, and the Consistency Ranking bars share one scale so they read in order. Chart hover labels now have a solid background, and the Standings Bump Chart renders again and seeds division winners first in division leagues. New charts: Floor / Ceiling, Luck Delta, Scoring Heatmap, Form: Last 3 Weeks, League Scoring by Week, and Career Wins by Team.",
        "link": "/graphs"
    },
    {
        "date": "2026-10-27",
        "tag": "update",
        "text": "My Leagues loads faster: the Rooting Guide and Bye Watch now answer from a cached result right away and refresh in the background instead of rescanning your leagues on every load, and Player Holdings is sortable: click any column header (position, player, rank, shares, bye, status, 7d trend, value) to sort the whole table.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-27",
        "tag": "update",
        "text": "My Leagues: the Rooting Guide and Bye Watch are cleaned up. Rooting Guide type and player rows are smaller and tighter, each game splits into two columns with your starters under the team they play for, a player you start in several leagues shows once with all of his league badges together, and upcoming games show the projection in muted type instead of a 0.0 for every player. Bye Watch now says Full strength once per league on a single line, with the leagues that have a hole to fill listed first.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-27",
        "tag": "fix",
        "text": "Front Office: the moves board no longer sits at Scanned 0 for accounts with many leagues. Every league the scan finishes now counts right away and its moves are kept, the scanned count climbs as the remaining leagues land, and the finished board is remembered. The board fills in over a few seconds instead of waiting for all of your leagues to be ready at the same moment.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-27",
        "tag": "fix",
        "text": "Trade Analyzer: the navy fair-range bar in the middle of the trade summary track is back. A site-wide tooltip update had quietly knocked it out of place, so the track showed only the marker even for trades the verdict called very fair.",
        "link": "/trade"
    },
    {
        "date": "2026-10-27",
        "tag": "fix",
        "text": "BR Assistant now follows the league you are actually in. It used to keep answering for an earlier league on pages like My Leagues, even while the league name at the top showed a different one: it now matches the league at the top of the page, and switching leagues in the app switches the assistant too.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-27",
        "tag": "fix",
        "text": "Weekly Recap alert dot: opening the recap from the navigation now clears its dot right away (it previously only cleared after a full page reload), and the Tuesday recap-ready notification now opens the recap itself instead of the matchups page.",
        "link": "/recap"
    },
    {
        "date": "2026-10-27",
        "tag": "update",
        "text": "Weekly Recap: the Injury Report is cleaned up so player names are never cut off, with the status pill beside the name and the fantasy team listed under it. The Standings and Power Rankings cards now stretch to the same height, whichever is taller. Every team and player name on the page opens its team or player card, including League Activity, Lineup Review and the Game of the Week look-ahead.",
        "link": "/recap"
    },
    {
        "date": "2026-10-27",
        "tag": "fix",
        "text": "My Leagues: the Rooting Guide and Bye Watch populate again. Bye Watch now reads each starter's bye from his team's schedule and covers the next week that has not kicked off yet, so it warns you about the byes you are actually facing. The Rooting Guide no longer gives up on a slow first load: it keeps retrying while the scan lands, the way the moves board already did.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-27",
        "tag": "new",
        "text": "Weekly Recap and Breakout Engine now carry a small red dot in the navigation when there is a new one you have not opened yet. The dot clears when you open it and comes back when the next week's recap or the next breakout set lands.",
        "link": "/recap"
    },
    {
        "date": "2026-10-27",
        "tag": "fix",
        "text": "Team cards and My Leagues cards: playoff odds now read at one decimal and only show 100% or 0% when a team has mathematically clinched a playoff spot or been eliminated. The Teams page team drawer now shows playoff odds for every team too, with the same rule.",
        "link": "/teams"
    },
    {
        "date": "2026-10-27",
        "tag": "new",
        "text": "Notification center: the bell at the top of the page is a real button now, and it opens your alerts. Notifications you were sent, like injury updates, recaps and scores, are kept newest first with links straight to the league, next to the latest site updates. Notification settings are grouped by how they arrive, quiet hours can hold non-live alerts overnight and send them with the morning digest, and your choices save to your account so they apply on every device. The settings menu also shows your account and plan at the top.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-27",
        "tag": "fix",
        "text": "My Leagues: a league card no longer jumps ahead to next week's preview while its final result is still due. When the week rolls over, the finished matchup stays on the card even if that league's data was loaded before the last game ended.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-27",
        "tag": "fix",
        "text": "Site reliability: pages and My Leagues load more consistently, and Refresh Data now kicks off the rebuild and updates each league as it lands instead of blocking on a full rebuild, so a slow league no longer stalls the whole site or leaves the refresh spinning.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-27",
        "tag": "update",
        "text": "Player card Advanced tab: the Trending section is now one verdict instead of a stack of separate alarms. Your main production number leads, the metrics behind it are grouped as the cause and the effect, and a Role check line shows your target, share and snap numbers even when they have not moved, so an efficiency dip reads differently from a role loss. Catch rate is labelled as a completion rate, and yards per target no longer carries a per-game unit.",
        "link": "/metrics"
    },
    {
        "date": "2026-10-26",
        "tag": "fix",
        "text": "Advanced Metrics graphs: zooming now zooms the axes instead of the whole picture. The ranges narrow and the tick marks step finer while titles, labels and dots stay the same size, so you can spread out a crowded cluster and read exactly where everyone sits. Drag to pan, Reset to return to the full graph, and downloads still export the full graph.",
        "link": "/metrics"
    },
    {
        "date": "2026-10-26",
        "tag": "update",
        "text": "Player game log: the ScoreZone tab in a player's card is now a timeline. Stat tiles up top, a chart of how the points built up play by play, and every play in game order with a running total. An intercepted pass now reads as an interception instead of an extra incomplete target, so the plays in the log match the targets in the stat line.",
        "link": "/scorezone"
    },
    {
        "date": "2026-10-26",
        "tag": "update",
        "text": "Schedule Rankings: the Schedule Assistant page is now a straight rankings table. Compare every team's remaining schedule in the League view, or switch to My Team to see your players in the same ranking order, with an All view for your full roster at once. Ease, Next 3 and Playoff ranks are side by side for every player.",
        "link": "/schedule"
    },
    {
        "date": "2026-10-26",
        "tag": "update",
        "text": "My Leagues: the tag on each league card now matches your Teams page. Dynasty leagues show their competitive window (Powerhouse, Contender, Win-Now, Crossroads and the rest, with a roster shape tag when one applies), and redraft leagues show Contend, Bubble or Long Shot, so a one-year league is never called a rebuild.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-25",
        "tag": "fix",
        "text": "ScoreZone: defensive touchdowns and interceptions now credit your DST, matchup names no longer cut off on phone hero cards, and live plays keep flowing if the play collector stalls: the page now detects a stale feed and restarts collection on its own. The TD alerts bar can be dismissed for the day (with undo) instead of forever, and dark mode greens match the rest of the site.",
        "link": "/scorezone"
    },
    {
        "date": "2026-10-24",
        "tag": "new",
        "text": "Teams: competitive windows rebuilt. A new Powerhouse tier sits above Contender for the league's clear top team, Holding Pattern splits into Crossroads and Treadmill, and team cards can carry a roster shape tag: Deep, Top-Heavy, Cliff Risk or Pick-Loaded. Roster grades are steadier too: age scores on a smooth curve instead of steps, and scoring percentiles use each format's own top players.",
        "link": "/teams"
    },
    {
        "date": "2026-10-19",
        "tag": "fix",
        "text": "My Leagues This week's moves now checks all of your linked leagues, not just the first eight, and My Leagues live matchup scores now refresh in the background instead of making your cards wait on a rebuild.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-19",
        "tag": "fix",
        "text": "My Leagues This week's moves no longer claims you are all caught up when it could not check some of your leagues: it now says how many leagues could not be checked yet, retries once while your leagues load, and includes start/bench swap alerts like the ones in your lineup-lock reminders.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-15",
        "tag": "new",
        "text": "BR Assistant answers more of your questions: ask for a younger, older, or cheaper replacement for any player, who a player's handcuff is, who to stash, who to drop, who to stream, what a player is worth in draft picks, and who you should shop from your roster.",
        "link": "/"
    },
    {
        "date": "2026-10-15",
        "tag": "fix",
        "text": "Fixed: shared page caches could show another manager's name tag. The Teams page now marks your team in your browser only, career graphs highlight your own team, and power rankings and assistant answers are cached per league.",
        "link": "/teams"
    },
    {
        "date": "2026-10-15",
        "tag": "fix",
        "text": "Fixed: the dashboard could ask signed-in managers to link their team again, even with the league already linked. Your weekly actions now load from your linked team on every visit.",
        "link": "/"
    },
    {
        "date": "2026-10-14",
        "tag": "new",
        "text": "Meet your BR Assistant: tap the BR button in the corner of your dashboard and ask about your team. Start or sit, trade checks, buy, sell or hold, waiver targets and a plan for the week, every answer built from your league data with the numbers to back it up.",
        "link": "/"
    },
    {
        "date": "2026-10-14",
        "tag": "new",
        "text": "Teams: dynasty leagues now have a Dynasty / This Season toggle. Switch to This Season and every roster is re-ranked on this-season (redraft) value instead of dynasty value, grades and position ranks included, so you can see who is built to win right now. Switch back to Dynasty for the long view.",
        "link": "/teams"
    },
    {
        "date": "2026-09-30",
        "tag": "new",
        "text": "Lineup Lab: a brand new lineup simulator on the Start/Sit tab. It plays out your matchup thousands of times to give you a real win probability, floor and ceiling ranges for every starter, the best bench swaps, and a why breakdown for each spot. Optimize for the highest win chance or chase upside when you need a ceiling game. Find it in the Weekly nav menu, right under Waivers & Start/Sit.",
        "link": "/waivers?tab=lab"
    },
    {
        "date": "2026-09-28",
        "tag": "new",
        "text": "ScoreZone Moments now live on the matchup page too: the Weekly Hub Matchups tab has the same TD and big-play cards with YOU/OPP filters and headshots, right at the top of the tab.",
        "link": "/"
    },
    {
        "date": "2026-09-24",
        "tag": "new",
        "text": "Front Office Report v2: the Season Hub report is now a structured, data-grounded team briefing instead of a prose-only memo. The card shows a verdict stamp, headline, key numbers, and your top move. The full report adds week-over-week changes, positional grades, trade targets with Trade Analyzer links, waiver targets, cut candidates, and a GM alert. Every number is computed, never invented.",
        "link": "/dashboard"
    },
    {
        "date": "2026-09-24",
        "tag": "new",
        "text": "Weekly Wrapped: the weekly hub now has story-style recap decks for your week, just like the Season Wrapped. Both can be shared with a public link, so league mates can watch without logging in.",
        "link": "/"
    },
    {
        "date": "2026-09-09",
        "tag": "fix",
        "text": "ESPN Start/Sit now reads your actual weekly lineup (including the QB slot) instead of treating missing ESPN slot ids as an empty lineup. The optimal-lineup banner compares against the same starters you see in the ESPN app.",
        "link": "/waivers?tab=startsit"
    },
    {
        "date": "2026-09-08",
        "tag": "fix",
        "text": "My Leagues This week's moves no longer disappears when the digest is empty. PRO users now always see the section: it shows an all-caught-up note when nothing needs attention across your leagues instead of hiding the card so it looks missing.",
        "link": "/portfolio"
    },
    {
        "date": "2026-09-08",
        "tag": "fix",
        "text": "Season Hub Actions tab no longer shows up blank. When your team is not linked yet it prompts you to link it, and when your lineup, roster, and waivers are all clean it says you're all set for the week instead of rendering nothing.",
        "link": "/dashboard"
    },
    {
        "date": "2026-09-08",
        "tag": "fix",
        "text": "Teams Schedule tab remaining-schedule ranking now uses the same SOS formula as standings (65% scoring + 35% win rate). Preseason uses projected starter scoring, then those projections fade in as a 4-game prior instead of dropping off after week 1.",
        "link": "/teams"
    },
    {
        "date": "2026-09-08",
        "tag": "fix",
        "text": "My Leagues position strength now matches the league Teams page, using the same starter-weighted ranker and the same values the player modal shows (1QB vs Superflex, redraft vs dynasty, PPR/TE premium).",
        "link": "/portfolio"
    },
    {
        "date": "2026-09-08",
        "tag": "update",
        "text": "Lineup-lock alerts now recommend concrete Sit A for B swaps even when a starter is hurt or on bye (up to two), and waiver pushes deep-link into Waivers with FAAB/drop guidance instead of the Players page.",
        "link": "/waivers"
    },
    {
        "date": "2026-09-08",
        "tag": "fix",
        "text": "Mobile polish: truncated trade chips and waiver rows expose the full name on hover, empty activity cards leave less blank space, and hub jump-nav / footer tap targets stay reachable.",
        "link": "/"
    },
    {
        "date": "2026-09-08",
        "tag": "fix",
        "text": "Draft Room Deep Dive treats rookie drafts as a class recap: no fake starters vs league, no Early tag after a team is done, no thin-roster or waiver flags, and playoff odds only on live connected leagues. Steal/reach cards, survival, and Decision Score no longer use startup lineup math on a 3-round board.",
        "link": "/draft"
    },
    {
        "date": "2026-09-08",
        "tag": "update",
        "text": "Weekly digest: if you have more than one connected league, Tuesday's email is a snapshot of all of them (record + one focus line each) with lineup/injury bullets on top, instead of a deep dive on the last league you opened plus a name-only list of the rest.",
        "link": "/portfolio"
    },
    {
        "date": "2026-09-08",
        "tag": "update",
        "text": "My Leagues This week's moves: waiver adds now require a startable rank or a real roster hole instead of always naming the highest leftover free agent. Deep RB35s, 1QB streamer QBs, and aging dynasty leftovers are dropped; cards explain the need or startable reason.",
        "link": "/portfolio"
    },
    {
        "date": "2026-09-06",
        "tag": "fix",
        "text": "Draft recommendations always rank the full undrafted pool for the pick on the clock. Rec no longer looks ahead to later owned picks or hides projected keepers, so early-pick advice stays early-round BPA in live, mock, and manual drafts.",
        "link": "/draft"
    },
    {
        "date": "2026-09-04",
        "tag": "update",
        "text": "Trade Targets: the list is mixed across your real holes (not four elites at each weak position). A quality starter at a bottom-ranked spot is no longer treated as a shop-for-Allen need; mid-tier players who fill the hole and owners who need your surplus rank first, with a one-line reason on each row.",
        "link": "/trade"
    },
    {
        "date": "2026-09-04",
        "tag": "update",
        "text": "Trade Targets: suggestions rank players by roster fit -- how much they upgrade your gap, whether you can pay, owner surplus, and age window -- instead of always listing the top few at a weak position.",
        "link": "/trade"
    },
    {
        "date": "2026-09-04",
        "tag": "fix",
        "text": "My Leagues: Signed in as now uses your Google name, not the last league's team owner. Opening an ESPN, Yahoo, or Fleaflicker league used to overwrite that label (and could attach the wrong Sleeper roster). Each card also shows your team name; Rank is regular-season standings (wins, then points for) for that team.",
        "link": "/portfolio"
    },
    {
        "date": "2026-09-03",
        "tag": "new",
        "text": "Advanced Metrics: pick more than one season (2025 + 2024 + 2022, or any years we have). The board defaults to one row per player-season (Year column on). Switch to Combine for one row per player (rates volume-weighted, totals added). Controls are grouped: metric + seasons + search, then positions, then filters (team, sort, age, volume). In a player modal, tap extra year pills to combine only seasons that player actually has; Career still uses every year.",
        "link": "/metrics"
    },
    {
        "date": "2026-09-01",
        "tag": "fix",
        "text": "Draft Cheat Sheet: changing PPR, TE premium, or passing-TD settings now updates Proj PPG and TE roster targets immediately. The compact board payload had dropped the scoring-variant map, so those dropdowns kept the previous scoring.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-09-01",
        "tag": "update",
        "text": "Draft Cheat Sheet: the Big Board now has Off Rk, the player's NFL team's projected offense rank (1 = best) from season-long implied totals. Same number Hist uses for projected-offense buckets. Sortable, in the CSV, and it does not change VOR order.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-09-01",
        "tag": "update",
        "text": "Draft Cheat Sheet Hist: the cohort count now says Samples: 6 instead of Sample: 6. That number is how many historical player-seasons are in the group.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-09-01",
        "tag": "fix",
        "text": "Draft Cheat Sheet Hist: tiny cells compile nested sibling buckets (no overlap) until the sample is at least 4, then fall back to the parent. Gibbs uses Round 1, age 23-24, last-year top-5 (n=6) instead of Sample 2. 32+ cells still wait for 15 so Kelce is not 4/4 of himself.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-09-01",
        "tag": "fix",
        "text": "Draft Cheat Sheet Hist: a 32+ TE coming off a top-5 year no longer headlines 2/2 as a 77% hit. That cell is often the same veteran repeating (Kelce). The percent uses other veteran TEs who were top-5 last year instead of counting this player's own seasons as comps.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-09-01",
        "tag": "fix",
        "text": "Draft Cheat Sheet Hist: a young RB1's tiny cell no longer borrows its chance from 30-year-old declining backs. The prior keeps the age band, so Gibbs sits in the mid-50s (in line with Round 1) instead of looking like a coin flip. Only 12 RBs finish top-12 each year; a typical RB is about 8%.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-09-01",
        "tag": "fix",
        "text": "Draft Cheat Sheet Hist: a tiny exact cell (Sample: 2) no longer shrinks a star toward every player at the position. The top-12 chance is pulled toward similar players (RBs coming off a top-5 year), so Gibbs stays in the 40s instead of dropping to 15%.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-09-01",
        "tag": "fix",
        "text": "Draft Cheat Sheet Hist: when a roster-split cell exists, the modal keeps that tile and leaves off the coarser same-band copy (21-32 projected offense, RB1, year 1 instead of also 21-32 projected offense, year 1).",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-09-01",
        "tag": "fix",
        "text": "Draft Cheat Sheet Hist: when a player is NFL Top 10 or Picks 11-25, the modal keeps that band and leaves off the parent Round 1 copy of the same roster or offense cell.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-09-01",
        "tag": "fix",
        "text": "Draft Cheat Sheet Hist: the player modal only shows buckets this player matches this year. Analog contrast (Top-10 offense when the team is 21-32, Round 1 RB3+), miss tiles, and last-year offense ranks are left off.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-09-01",
        "tag": "fix",
        "text": "Draft Cheat Sheet Hist: the board Hist column uses the same top-12 chance as the player modal. Duplicate year-1 copies, the middle roster analog (RB2/WR2), and the miss tile are left off so the matching cell is easier to read.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-09-01",
        "tag": "fix",
        "text": "Draft Cheat Sheet Hist: on phones the tiles stack in one column so titles, This player, and names are not squeezed into a two-column grid.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-09-01",
        "tag": "update",
        "text": "Draft Cheat Sheet Hist: tiles are grouped (Career, Capital, Roster, Offense, Usage), the matching cell is marked This player versus analog contrast, and each tile can open 2-3 names from that intersection. Bounce-back WRs and RBs who were outside the top 36 last year are split by roster spot.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-09-01",
        "tag": "update",
        "text": "Draft Cheat Sheet Hist: NFL Round 1 rates now split by preseason roster spot for RB, WR, and QB (a Round 1 WR1 is not mixed with a Round 1 WR3+), and RB Round 1 / Top 10 pick rates are shown inside each projected-offense band so a first-round back on a 21-32 offense is not mixed with the whole room. Intersections are counted, not multiplied.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-09-01",
        "tag": "update",
        "text": "Draft Cheat Sheet Hist: projected-offense rates now split by preseason roster spot (RB1, WR1, TE1, QB1 vs 2 vs 3+ by ADP among teammates), so a starter on a top-10 projected offense is not mixed with an RB3, WR3, or backup QB.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-09-01",
        "tag": "update",
        "text": "Draft Cheat Sheet Hist: sample sizes now say Sample: 318 instead of n=318.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-09-01",
        "tag": "update",
        "text": "Draft Cheat Sheet Hist: team-offense rows now use season-long implied totals from nflverse (spread and total on regular-season games) as the projected offense, cover 2016-2025, and still show last year's actual rank as a second analog.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-09-01",
        "tag": "update",
        "text": "Draft Cheat Sheet Hist: capital rows now say the year (Drafted NFL Top 10, year 1 vs any season), keep small exact cells instead of dropping draft capital, and show the drafted-band x career-year hit rate.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-08-31",
        "tag": "update",
        "text": "Dashboard: Matchup Preview is the scoreboard and win bar only. Open the title to see the full matchup page.",
        "link": "/"
    },
    {
        "date": "2026-08-31",
        "tag": "update",
        "text": "Dashboard: Front Office Report no longer shows a team-name chip under the title.",
        "link": "/"
    },
    {
        "date": "2026-08-31",
        "tag": "update",
        "text": "Dashboard: Front Office Report sits under Standings in the left column so the hub is less lopsided, and league bulletins are off on every platform.",
        "link": "/"
    },
    {
        "date": "2026-08-31",
        "tag": "fix",
        "text": "Dashboard: Matchup Preview stays hidden until the league drafts, except in dynasty. Keeper Fleaflicker leagues no longer show last year's matchups before the new draft.",
        "link": "/"
    },
    {
        "date": "2026-08-31",
        "tag": "update",
        "text": "Onboarding: shorter site tour with mobile steps, Remind me later, interactive spotlights, and a Premium welcome you can replay from Settings. Tour resume no longer swaps in demo data on History.",
    },
    {
        "date": "2026-08-28",
        "tag": "update",
        "text": "Draft Room: Cheat Sheet is a one-tap control next to Undo and Trade on the live/mock board, instead of living in the Settings dropdown.",
        "link": "/draft"
    },
    {
        "date": "2026-08-28",
        "tag": "update",
        "text": "Redraft Cheat Sheet: the board starts loading immediately and uses a smaller player payload, so the first paint is faster.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-08-28",
        "tag": "fix",
        "text": "Dashboard: leagues that have not drafted yet replace the empty roster sidebar with a Draft Cheat Sheet preview and a link to the full board.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-08-28",
        "tag": "fix",
        "text": "My Leagues: leagues that have not drafted (empty roster shells) now show a draft countdown instead of fake positional ranks.",
        "link": "/portfolio"
    },
    {
        "date": "2026-08-28",
        "tag": "fix",
        "text": "Draft Cheat Sheet Hist: the modal now shows two groups (players like this vs that ADP round) instead of a vs-market edge, so 37% and 57% are not read as one combined chance.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-08-28",
        "tag": "fix",
        "text": "Draft Room Deep Dive: Hist vs ADP is labeled as two groups, and a lower Hist than the ADP round is not painted as a miss.",
        "link": "/draft"
    },
    {
        "date": "2026-08-28",
        "tag": "update",
        "text": "Draft Room Deep Dive: redraft reports now include historical top-12 trends for your skill picks versus the ADP-bucket hit rate. Descriptive only. Not a ranking or Pick Score input.",
        "link": "/draft"
    },
    {
        "date": "2026-08-28",
        "tag": "fix",
        "text": "Draft Cheat Sheet: on a phone, Rk and Player stay pinned while the rest of the big board scrolls sideways.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-08-28",
        "tag": "fix",
        "text": "Draft Cheat Sheet Hist: closest-example tags now read Exp: Year 4, Draft: Round 1, Age: 23-24, and Last Year: Top 5.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-08-28",
        "tag": "fix",
        "text": "Draft Cheat Sheet Hist: closest-example tags now say NFL year, NFL Round 1, age, and last year Top 5, so they are not mixed up with the season year or this season's finish.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-08-28",
        "tag": "update",
        "text": "Draft Cheat Sheet Trends: the sticky profile and matching-player dock can be collapsed so the tables stay in view. Lane chips stay on screen.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-08-28",
        "tag": "update",
        "text": "Draft Cheat Sheet Hist: closest examples now mark Top-5, Top-12, and Top-24 hits on each row and in the summary, so a #1 is a Top 5 and a #20 is a Top 24.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-08-28",
        "tag": "fix",
        "text": "Draft Cheat Sheet Trends: collapsed tables on a phone stack the heading and the peek line so NFL Round 1 52% is not clipped off the right edge.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-08-28",
        "tag": "fix",
        "text": "Draft Cheat Sheet Hist: closest-example tags now read Year 6+, Round 1, and Top 5 instead of underscored keys.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-08-28",
        "tag": "update",
        "text": "Draft Cheat Sheet Trends: NFL Round 1 capital is split into Top 10, picks 11-25, and the rest of Round 1, so early and late first-rounders are not mixed. Still NFL draft capital, not fantasy ADP.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-08-28",
        "tag": "fix",
        "text": "Draft Cheat Sheet Trends: combining a named bucket (NFL Round 1) with a numeric one (never top-12) no longer fails to load the selected profile.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-08-28",
        "tag": "update",
        "text": "Draft Cheat Sheet Trends: the selected profile is a compact verdict card with hit-rate tiles, edge chips, and a sample-size pill, so the mix stays readable without covering the tables.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-08-28",
        "tag": "fix",
        "text": "Draft Cheat Sheet Trends: the selected-bucket dock is compact so the tables stay on screen, and this year's rookies (including 1st-rounders with no NFL season yet) show in matching-player lists from their NFL draft capital.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-08-28",
        "tag": "update",
        "text": "Draft Cheat Sheet Trends: selected buckets now return the true combined historical hit rate from matching player-seasons (not multiplied odds), with shrinkage-adjusted edges, Wilson intervals, market-adjusted Top-12 when ADP coverage is solid, Scout match reasons, and closest historical examples in Hist. Historical data stays descriptive-only.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-08-28",
        "tag": "fix",
        "text": "Draft Cheat Sheet Trends: the matching-player dock sits below the site nav instead of sliding under it.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-08-28",
        "tag": "fix",
        "text": "Draft Cheat Sheet Trends: matching board players sit in a sticky dock above the tables, so tapping a bucket lists names in place instead of rebuilding the page. Browsing stays free; the full matching list is still the existing Pro scout preview.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-08-28",
        "tag": "update",
        "text": "Draft Cheat Sheet Trends: last-year volume tables for RB touches (including the 400+ workhorse cliff) and carries, WR/TE receptions and targets, games played, and QB pass attempts. Same descriptive hit rates as the other usage slices.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-08-28",
        "tag": "fix",
        "text": "Draft Cheat Sheet Trends: matching board players sit below the tables in two columns so tapping a bucket does not send you back to the top. Last-year RYOE and aDOT filters now list current-board players who actually have those stats.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-08-28",
        "tag": "fix",
        "text": "Draft Cheat Sheet Hist: the headline percent is this player's historical chance given his career (already top-12 or not) and current situation (year, capital, last year), not last-year lookalikes alone.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-08-28",
        "tag": "fix",
        "text": "Draft Cheat Sheet Hist: a down year no longer reads as never top-12. First-time elite uses every observed season, not just last year's finish. Breakout and smash still follow last year.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-08-28",
        "tag": "fix",
        "text": "Draft Cheat Sheet Hist: capital and age rows no longer repeat the same bucket under one title. Miss rate, rookie, and year-2 rates keep their own labels, and exact age drops when the age band is already shown.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-08-28",
        "tag": "update",
        "text": "Draft Cheat Sheet Trends: tap buckets to list current-board players who match (AND across tables). Finish chips show typical top-5, top-12, and top-24 rates. Browsing stays free; the matching player list is a Pro scout with a short preview.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-08-28",
        "tag": "new",
        "text": "Draft Cheat Sheet: Trends tab and Hist player cards. Trends shows position-wide historical top-12 rates (ADP, career, capital, age, usage) versus a typical player at that position. Hist opens a player's own mix of those buckets.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-08-28",
        "tag": "fix",
        "text": "My Leagues positional strength now shows your average percentile at each position across leagues (50th is a typical team) instead of a signed percent vs league median, so a stacked finish in one league is no longer dragged negative by a thin one.",
        "link": "/portfolio"
    },
    {
        "date": "2026-08-26",
        "tag": "fix",
        "text": "Draft Room recommendations for your first pick no longer label every player Best available: while you wait, Rec ranks who is actually likely to be there (not 1.01 talent at 0%), and Best available is reserved for the top rec rather than used as a generic fallback.",
        "link": "/draft"
    },
    {
        "date": "2026-08-26",
        "tag": "fix",
        "text": "Draft Room recommendations no longer let an empty 1QB or 1TE slot leap remaining WR/RB depth: filling that one streamable starter is still a real need, but it now competes with skill-position depth instead of ranking a board of QBs and TEs over a WR run.",
        "link": "/draft"
    },
    {
        "date": "2026-08-26",
        "tag": "fix",
        "text": "Draft Room Deep Dive no longer labels leftover-ADP as a reach: taking the best remaining player (or anyone in a tight ADP cluster) is Fair even if historical ADP sits a couple of spots later, and so is anyone with under 20% chance to last to your next pick.",
        "link": "/draft"
    },
    {
        "date": "2026-08-25",
        "tag": "fix",
        "text": "Player modal Trades tab: This League now looks across every season in the league's history (including offseason trades), shows real team names on each side, and aligns the two packages toward the swap.",
        "link": "/players"
    },
    {
        "date": "2026-08-25",
        "tag": "fix",
        "text": "Start/Sit Advisor: lineup swaps now pair like-for-like (QB for QB, with FLEX/SUPERFLEX labeled separately) instead of mixing positions, and a negative gain no longer renders as +-.",
        "link": "/waivers"
    },
    {
        "date": "2026-08-24",
        "tag": "new",
        "text": "ADP Sources: Compare Sleeper, BR Fantasy, ESPN, MFL, and Yahoo ADP side by side in Player Rankings. Player modals show every available source on one range, with exact ADP values on hover and a Consensus built from all available sources.",
        "link": "/players"
    },
    {
        "date": "2026-08-12",
        "tag": "new",
        "text": "Draft Cheat Sheet: a printable, value-based draft board tuned to your league scoring, roster, and format, with players grouped into tiers. Open it during a live Sleeper draft or a mock from the Draft Room and it crosses off who is already gone.",
        "link": "/draft/cheat-sheet"
    },
    {
        "date": "2026-08-06",
        "tag": "new",
        "text": "Even It Out: When a deal favors one side, the Trade Calculator now suggests up to three players whose value best closes the gap. Click one to drop it onto the light side.",
        "link": "/trade"
    },
    {
        "date": "2026-07-13",
        "tag": "new",
        "text": "Start/Sit Advisor: every player now shows a boom-or-bust profile, with a floor-to-ceiling scoring range and a Steady, Balanced, Volatile, or Boom-or-bust read built from their weekly game-by-game scores (last season blends in until this year's sample fills out). Rows also surface live game conditions: the Vegas implied team total and a weather or venue chip (dome, cold, wind, rain). And the head-to-head Compare was rebuilt with one shared label per stat plus all of the new signals.",
        "link": "/waivers"
    },
    {
        "date": "2026-07-13",
        "tag": "new",
        "text": "Playoff Picture: the Playoff Odds now include an exact clinch and elimination outlook down the stretch. See who has clinched a berth or a first-round bye, who is mathematically eliminated (their row is struck through), who controls their own destiny, and the magic number of wins each team needs to clinch. It appears over the final five weeks or so of the regular season.",
    },
    {
        "date": "2026-06-24",
        "tag": "new",
        "text": "Draft Room: A full draft assistant with real-time pick tracking, best-available rankings, pick scoring, and post-draft grades. Run a mock for any format or connect directly to your live Sleeper draft. Supports rookie, startup, and redraft leagues with format-specific grading, team archetype detection, and a shareable draft summary card.",
        "link": "/draft"
    },
    {
        "date": "2026-06-18",
        "tag": "update",
        "text": "Player & Compare modals: metric bars now scale by positional rank, so bar length reflects how a player stacks up at their position (Total TDs, WOPR, Air Yards Share, Targets/G, Touches/G and more no longer render misleadingly). Game logs now load in the Compare view, and graphs reopen instantly thanks to client-side caching.",
        "link": "/metrics"
    },
    {
        "date": "2026-06-17",
        "tag": "new",
        "text": "Rich link previews: Sharing a metrics graph, a trade card, or a team report card now unfurls with a real preview image of that exact card, generated on the server, instead of just the BR logo, on iMessage, Discord, X, and more.",
        "link": "/metrics"
    },
    {
        "date": "2026-06-17",
        "tag": "new",
        "text": "Advanced Metrics Graph: Plot any two (or three) efficiency metrics as a scatter chart directly from the Advanced Metrics page. Includes position presets for WR, RB, and TE, bubble sizing by a third stat, top-player labels, average crosshair lines, light/dark mode, and a BR watermark. Download the chart as a PNG or copy a shareable link that reopens the exact graph configuration.",
        "link": "/metrics"
    },
    {
        "date": "2026-06-13",
        "tag": "new",
        "text": "Player Modal: Advanced Metrics tab now shows positional ranks for every efficiency stat with position-group minimums (4+ games, 20+ carries, 15+ targets, 50+ pass attempts). Weekly Trends is here: see a sparkline, rolling average, and trend arrow for key stats across the season with a time filter to zoom in on any stretch of weeks.",
        "link": "/players"
    },
    {
        "date": "2026-06-11",
        "tag": "update",
        "text": "We've moved to brfantasyfootball.com. Same app, new home. Update your bookmarks!"
    },
    {
        "date": "2026-06-10",
        "tag": "feature",
        "text": "Advanced Metrics: Sort by any column, filter by team, and compare up to 4 metrics side by side. Pin players to keep them at the top of the leaderboard. Year-over-year trend arrows show at a glance whether a player improved or declined from last season.",
        "link": "/metrics"
    },
    {
        "date": "2026-06-09",
        "tag": "new",
        "text": "Share Card: Generate a shareable report card for any team in your league. Shows team grade, dynasty rank, dynasty value with picks, positional rankings, top players with ages, draft capital, and average roster age. Copy the link and send it to anyone. Find it under each team on the Teams page.",
        "link": "/teams"
    },
    {
        "date": "2026-06-08",
        "tag": "new",
        "text": "Advanced Metrics Leaderboard: See every player ranked by a single advanced metric: role score, snap share, target share, yards per carry, air yards, and more. Filter by position, search by name, and switch between seasons. Find it under the Players menu.",
        "link": "/metrics"
    },
    {
        "date": "2026-06-08",
        "tag": "new",
        "text": "Roster Filter: The Trade Calculator now lets you search from actual rosters. Enable Roster Filter to restrict each side's search to that team's real players and picks, auto-binding Team 2 the moment you add your first opponent asset.",
        "link": "/trade"
    },
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
        "text": "Player Scoring: Player modals now show PPG and season total points side by side (e.g. 24.5 | 416.5) with positional ranks for both: PPG · RB1 | TOTAL · RB2. Compare player view shows the same scoring breakdown for both players.",
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
        "text": "Team Reports: the AI-powered Front Office Report analyzes your roster, trade opportunities, and league standings",
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
