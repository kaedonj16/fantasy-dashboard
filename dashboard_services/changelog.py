"""
Changelog entries for the fantasy dashboard.
Each entry represents a user-facing change.
"""

CHANGELOG = [
    {
        "date": "2026-11-25",
        "tag": "update",
        "text": "On phones, the league dashboard now has a Jump to bar and a Tools grid, so your matchup, waivers, trades and more are one tap away instead of a long scroll. Standings, activity and awards start tucked away and expand when you want them.",
        "link": "/dashboard"
    },
    {
        "date": "2026-11-25",
        "tag": "update",
        "text": "Top role jobs got stricter: Alpha WR1, Workhorse, and Alpha TE1 now take a higher score, and only the winner of his own room can hold one. Rooms are judged on per-game workload, so a player who missed time with an injury keeps the title his role earned. Franchise QB1 counts efficiency as well as volume, so the title goes to quarterbacks who produce, not just the ones who throw the most.",
        "link": "/compare"
    },
    {
        "date": "2026-11-24",
        "tag": "new",
        "text": "Prospect boards now go back through history: pick any draft class from 2016 on, or All, to see every class's grades and rankings, with past classes graded by the current model.",
        "link": "/prospects"
    },
    {
        "date": "2026-11-24",
        "tag": "update",
        "text": "The Breakout Engine board now splits its calls into Breaking Out and Confirming: Breaking Out flags players whose role is jumping before the points arrive, and Confirming tracks players already producing whose new role the engine expects to hold.",
        "link": "/breakouts"
    },
    {
        "date": "2026-11-24",
        "tag": "update",
        "text": "Prospect profiles are now a single report card: grade, projected NFL role, production chart, season lines, metric percentiles, comparables, and scouting notes on one scroll, no tabs.",
        "link": "/prospects"
    },
    {
        "date": "2026-11-23",
        "tag": "update",
        "text": "Prospect grades are back on the rookie board: every prospect now shows his numeric grade, on the board and beside his tag when you open his profile. The board's tag chips are gone; the tier lines name each group instead.",
        "link": "/prospects"
    },
    {
        "date": "2026-11-21",
        "tag": "update",
        "text": "Rest of Season rankings are rebuilt on a leaner model, so the order tracks projected production more closely.",
        "link": "/rankings/rest-of-season"
    },
    {
        "date": "2026-11-21",
        "tag": "update",
        "text": "Rest of Season overall rankings now value 1QB quarterbacks against the best option on waivers, so QBs slot where their edge actually puts them.",
        "link": "/rankings/rest-of-season"
    },
    {
        "date": "2026-11-20",
        "tag": "update",
        "text": "QB role jobs now count rushing: a dual-threat's legs count toward Franchise QB1, and the top title needs at least 5 games in the window.",
        "link": "/compare"
    },
    {
        "date": "2026-11-20",
        "tag": "fix",
        "text": "Prospect percentiles no longer pile up at 100: they're ranked on the raw stats now, so top prospects separate properly.",
        "link": "/prospects"
    },
    {
        "date": "2026-11-19",
        "tag": "update",
        "text": "Role jobs now respect the depth chart: a player who owns his room earns the top job title even when his raw share trails the league's bell cows.",
        "link": "/compare"
    },
    {
        "date": "2026-11-19",
        "tag": "new",
        "text": "Start/Sit, Player Rankings and Rest of Season rows now show each player's role job next to his name, with a Contingent note when his production depends on an injured roommate.",
        "link": "/rankings/rest-of-season"
    },
    {
        "date": "2026-11-19",
        "tag": "new",
        "text": "Compare now reads roles, not just stats: every player gets a job title, role axes, a target map, a carry gap map, a role timeline, and his closest role matches across the league.",
        "link": "/compare"
    },
    {
        "date": "2026-11-18",
        "tag": "new",
        "text": "The home page is rebuilt around your Sunday: the live values ticker sits on top, your matchup, start/sit, ScoreZone, trade and Wrapped previews live in one tappable gallery, and connecting a league starts with one field.",
        "link": "/"
    },
    {
        "date": "2026-11-16",
        "tag": "update",
        "text": "Playoff Path root-for games and your matchup now show live scores during games and final scores when they end, refreshing on their own while a game you track is live.",
        "link": "/standings"
    },
    {
        "date": "2026-11-15",
        "tag": "update",
        "text": "ScoreZone now feels live: scores count up and flash in team colors, plays land with a pop, the field board deepens as a drive pushes on, and a condensed score bar sticks while you scroll.",
        "link": "/scorezone"
    },
    {
        "date": "2026-11-11",
        "tag": "new",
        "text": "Season Wrapped is rebuilt as an 11-slide story: the champion, their Road to the Title, league MVP, season records, the Hardware awards, final standings, and a Your Season slide built for your team.",
        "link": "/history"
    },
    {
        "date": "2026-11-11",
        "tag": "update",
        "text": "Power Rankings now read as a Power Index from 1 to 99: the best team in your league sits in the 90s and a league-average team is a 50. Same order, just a score that's easier to follow week to week.",
        "link": "/standings"
    },
    {
        "date": "2026-11-10",
        "tag": "update",
        "text": "Playoff Path's Root For list now shows how each game also moves your bye and 1 seed odds, next to the playoff swing.",
        "link": "/standings"
    },
    {
        "date": "2026-11-10",
        "tag": "update",
        "text": "Playoff Path now shows up where you make decisions: the Front Office Report and the GM Brief both carry your clinch math and this week's stakes in the final weeks of the season.",
        "link": "/"
    },
    {
        "date": "2026-11-10",
        "tag": "update",
        "text": "The BR Assistant now answers Playoff Path questions: who to root for, how your playoff, bye and 1 seed odds move if you win or lose, and where you stand on tiebreakers.",
        "link": "/standings"
    },
    {
        "date": "2026-11-10",
        "tag": "update",
        "text": "Standings on mobile is now tabbed with a compact league summary, and Playoff Impact can switch between playoff, bye and 1 seed odds.",
        "link": "/standings"
    },
    {
        "date": "2026-11-10",
        "tag": "new",
        "text": "Weekly Wrapped adds a Playoff Picture slide down the stretch, calling out clinched and eliminated teams. The whole deck also gets a fresh look and plays full screen on phones.",
        "link": "/weekly"
    },
    {
        "date": "2026-11-10",
        "tag": "new",
        "text": "New Playoff Path tab on Standings: your magic number and earliest clinch, the games to root for this week, tiebreakers, and how every team's odds swing on a win or a loss.",
        "link": "/standings"
    },
    {
        "date": "2026-11-09",
        "tag": "update",
        "text": "The Rooting Guide on My Leagues loads much faster, and it now leads with the math: how many points you need to flip a game, how many your opponent needs, and when a lead is locked up.",
        "link": "/portfolio"
    },
    {
        "date": "2026-11-09",
        "tag": "new",
        "text": "Player Rankings now has a Rest of Season mode: every player ranked by projected production for the weeks left, with a Window toggle that splits the board into the full rest of season, the stretch run through Week 14, and the fantasy playoffs in Weeks 15 to 17.",
        "link": "/players"
    },
    {
        "date": "2026-11-08",
        "tag": "new",
        "text": "Search on mobile no longer opens to a blank page: it suggests your recent searches, top players, and shortcuts before you type. The page also no longer zooms in when you tap a text field on your phone.",
        "link": "/"
    },
    {
        "date": "2026-11-08",
        "tag": "new",
        "text": "Matchups: tapping a live drive bar now opens the game right there, with the full board, a box score Stats tab, and a Plays tab, instead of leaving for ScoreZone.",
        "link": "/weekly"
    },
    {
        "date": "2026-11-07",
        "tag": "update",
        "text": "The mobile More menu is rebuilt: it glides open, sections show what's inside them, your current page is named on its section, and Standings, Start/Sit, Player Rankings and ScoreZone sit in a new Quick group.",
        "link": "/"
    },
    {
        "date": "2026-11-06",
        "tag": "update",
        "text": "Advanced Metrics: Last 2 and Last 4 now mean each player's last games played, not the last calendar weeks, so byes and missed games no longer shrink the sample.",
        "link": "/advanced-metrics"
    },
    {
        "date": "2026-11-06",
        "tag": "new",
        "text": "Draft Database is here: browse completed Sleeper drafts from across the site, with full boards, biggest steals and busts, and ADP by segment.",
        "link": "/draft-database"
    },
    {
        "date": "2026-11-06",
        "tag": "update",
        "text": "Draft Room: the draft board is rebuilt with compact rows and real player headshots, so the board reads faster while your draft is live.",
        "link": "/draft"
    },
    {
        "date": "2026-11-05",
        "tag": "update",
        "text": "Advanced Metrics: the filters you reach for most now sit up front on the main bar, every set filter shows as a chip you can change or clear, and an empty board now says which filters caused it.",
        "link": "/advanced-metrics"
    },
    {
        "date": "2026-11-03",
        "tag": "new",
        "text": "Championship odds are here: a new Title column on Playoff Odds shows each team's chance to win it all, and the same number joins the Season Hub and team modal down the stretch.",
        "link": "/standings"
    },
    {
        "date": "2026-11-03",
        "tag": "update",
        "text": "Weekly Hub: the header and Week Leaders now form one panel on desktop, and the four leader cards have a cleaner, more uniform design.",
        "link": "/weekly"
    },
    {
        "date": "2026-11-03",
        "tag": "fix",
        "text": "Two playoff fixes for Fleaflicker and MFL leagues: the Playoff Picture now shows the full playoff field, and Playoff Odds can no longer slip into a made-up preseason projection mid-season.",
        "link": "/standings"
    },
    {
        "date": "2026-11-03",
        "tag": "fix",
        "text": "Prospect grading comparisons got cleaner: historical marks now come only from seasons where the player held a real role, and one freak season no longer stretches the comparison scale.",
        "link": "/prospects"
    },
    {
        "date": "2026-11-03",
        "tag": "update",
        "text": "Weekly Hub on phones: the matchup score header now sits on a white card of its own, matching the board below it, so the whole matchup stack reads as clean cards all the way down.",
        "link": "/weekly"
    },
    {
        "date": "2026-11-03",
        "tag": "update",
        "text": "Weekly Hub on phones: the site header is back at the top, Turning Points is a slim full-width card, and the header fits ScoreZone, Weekly Wrapped and the week selector on one line.",
        "link": "/weekly"
    },
    {
        "date": "2026-11-03",
        "tag": "new",
        "text": "Tuesday mornings now bring two combined pushes across your leagues: a 9am weekly results roundup and an 11am recap-ready notification. Quiet hours now end at 9am by default.",
        "link": "/portfolio"
    },
    {
        "date": "2026-11-03",
        "tag": "fix",
        "text": "Matchup win odds now respect the endgame: late in a game they're priced from the players actually still out there, and a QB and his pass catchers now move the odds together.",
        "link": "/weekly"
    },
    {
        "date": "2026-11-03",
        "tag": "fix",
        "text": "Reopening the app now refreshes your league data on its own: coming back after a minute or more away silently swaps in current data instead of sitting on a stale page.",
        "link": "/"
    },
    {
        "date": "2026-11-02",
        "tag": "fix",
        "text": "Weekly Hub on phones: swiping between matchups now shows one full matchup at a time, the head-to-head record sits on its own line under the team records, and board rows are tighter.",
        "link": "/weekly"
    },
    {
        "date": "2026-11-02",
        "tag": "update",
        "text": "Navigation tidied up: the Weekly menu is grouped into Play and Review, ScoreZone sits second where game day needs it, and NFL Team Rankings moved to the Stats menu.",
        "link": "/"
    },
    {
        "date": "2026-11-02",
        "tag": "new",
        "text": "Weekly Hub on phones is rebuilt compact: your matchup opens under a slimmer header with one row of five tabs, swipe-only previews, tighter board rows, and a slimmer bottom dock site-wide.",
        "link": "/weekly"
    },
    {
        "date": "2026-11-02",
        "tag": "fix",
        "text": "The app now updates itself when you open it: a new version swaps in on its own instead of needing a manual refresh. An update that lands mid-page still waits for your tap.",
        "link": "/"
    },
    {
        "date": "2026-11-02",
        "tag": "fix",
        "text": "Lineup lock reminders no longer tell you to bench a starter whose game already kicked off, and a locked starter's injury is no longer raised as something to fix.",
        "link": "/"
    },
    {
        "date": "2026-11-02",
        "tag": "new",
        "text": "Weekly Hub Scorers is rebuilt as one clean panel: every position on the same grid, real headshots, and the top scorer at each position stepping up a size.",
        "link": "/"
    },
    {
        "date": "2026-11-02",
        "tag": "fix",
        "text": "Matchup board fixes: a starter who played and got hurt mid-game no longer shows as BYE, so his real points stay on the board, and the position chip label is properly centered.",
        "link": "/"
    },
    {
        "date": "2026-11-02",
        "tag": "fix",
        "text": "Prospect percentiles are now graded only against drafted players from the 2016 to 2024 classes, and the stat percentiles that used to show a dash now fill in with real numbers.",
        "link": "/prospects"
    },
    {
        "date": "2026-11-02",
        "tag": "fix",
        "text": "Two polish fixes: the Player Rankings hover trend graph now matches the new value history charts, and player card game tooltips no longer list stat lines a player never touched.",
        "link": "/players"
    },
    {
        "date": "2026-11-02",
        "tag": "new",
        "text": "Every league graph is rebuilt in the site's own chart style: the team radar is now bars against the league average, every team keeps one fixed color on every chart, and every point can be hovered or tapped for its exact numbers.",
        "link": "/graphs"
    },
    {
        "date": "2026-11-02",
        "tag": "new",
        "text": "Value History charts are rebuilt in the site's own style: navy main line with a gradient under it, Superflex as a dashed sky line, Peak and Low marked with dates, and lines that pass through every real value exactly.",
        "link": "/players"
    },
    {
        "date": "2026-11-02",
        "tag": "fix",
        "text": "Player Rankings hover quick look: the 30-day trend graph now keeps its proportions instead of stretching to fill its box, so gentle moves no longer look like cliffs.",
        "link": "/players"
    },
    {
        "date": "2026-11-02",
        "tag": "fix",
        "text": "Player Rankings: Total Points and PPG now count each player's own finished games as soon as they end, matching the player card, and the hover quick look's sections time out and retry cleanly instead of spinning forever.",
        "link": "/players"
    },
    {
        "date": "2026-11-01",
        "tag": "fix",
        "text": "Trade Calculator: each side's header is now just the team name, player headshots on chips sit clean with no colored backing, and the search boxes have a cleaner look with a search icon.",
        "link": "/trade"
    },
    {
        "date": "2026-11-01",
        "tag": "fix",
        "text": "Trade Calculator: the Trade Fit grade now grades the trade from your side, and the acceptance read weighs the other team's window, so picks count for more to a rebuilder and proven players to a contender.",
        "link": "/trade"
    },
    {
        "date": "2026-11-01",
        "tag": "fix",
        "text": "Trade Calculator: player chips had the name and value flipped on phones. Chips now read headshot, name, then value, chip values always match the Trade Summary totals, and a divider separates the two sides.",
        "link": "/trade"
    },
    {
        "date": "2026-11-01",
        "tag": "fix",
        "text": "Closed the last gaps in the app's memory limit on My Leagues: refreshing many leagues at once can no longer stack heavy rebuilds until the app restarts. Anything over the limit waits for the next refresh.",
        "link": "/portfolio"
    },
    {
        "date": "2026-11-01",
        "tag": "fix",
        "text": "Fixed the app restarting under load: heavy background work now shares one strict per-worker limit, so opening My Leagues with many leagues can't pile up builds and exhaust memory.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-31",
        "tag": "new",
        "text": "Trade Calculator on phones: a slimmer header, flat full-width sides with running totals, and player chips that show headshots on phone and desktop.",
        "link": "/trade"
    },
    {
        "date": "2026-10-31",
        "tag": "fix",
        "text": "Trade Calculator: tapping the pinned balance bar now actually jumps to the Trade Summary. The tap was aiming at the wrong scroller, so it landed nowhere.",
        "link": "/trade"
    },
    {
        "date": "2026-10-31",
        "tag": "fix",
        "text": "Trade Calculator: the pinned bar is now a slim balance strip with just the two side values and the fair range. Tap it to jump to the full Trade Summary.",
        "link": "/trade"
    },
    {
        "date": "2026-10-31",
        "tag": "fix",
        "text": "Trade Calculator on phones: the pinned balance bar now ends before the BR Assistant button, so the totals, Trade Fit grade and verdict stay fully visible.",
        "link": "/trade"
    },
    {
        "date": "2026-10-31",
        "tag": "new",
        "text": "Pull to refresh now works on more pages in the app: My Leagues, Waivers, Standings and Player Rankings join ScoreZone and the Weekly Hub. Your filters, week and scroll position stay put.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-31",
        "tag": "fix",
        "text": "Fixed the app sending you back to the dashboard after you tapped into a page: a background refresh now belongs to the page it started on, and navigating away cancels it.",
        "link": "/dashboard"
    },
    {
        "date": "2026-10-30",
        "tag": "fix",
        "text": "Trade Calculator on phones: the pinned balance bar now floats above the bottom tab bar, and the Trade Fit grade shows a grading state while a new package is scored instead of the previous trade's number.",
        "link": "/trade"
    },
    {
        "date": "2026-10-30",
        "tag": "new",
        "text": "Trade Calculator: the summary now grades the deal with a Trade Fit score from 0 to 10, adds a Your Lineup After block under Playoff Impact, and offers a one-tap Counter card when your side is light.",
        "link": "/trade"
    },
    {
        "date": "2026-10-30",
        "tag": "fix",
        "text": "Teams: with a team's detail drawer open, the floating pills and BR Assistant button now sit behind the drawer instead of covering player values.",
        "link": "/teams"
    },
    {
        "date": "2026-10-30",
        "tag": "fix",
        "text": "Trade calculator consistency: the Playoff Impact verdict, the Trade Analyst's tier language, and the calculator's rows now all tell the same story as the summary and cards.",
        "link": "/trade"
    },
    {
        "date": "2026-10-30",
        "tag": "fix",
        "text": "Player Rankings loads in one pass now: the data request is less than half its old size, and tiers, trends and hover quick look are all there when the board settles.",
        "link": "/players"
    },
    {
        "date": "2026-10-30",
        "tag": "new",
        "text": "Guides: a Latest strip now leads the index with the three newest articles, and category chips stick at the top. The first Trade Case Study is live too.",
        "link": "/guides"
    },
    {
        "date": "2026-10-30",
        "tag": "update",
        "text": "Dynasty values: the market blend now uses KeepTradeCut as its second source alongside FantasyCalc, so the board tracks the live dynasty market more closely.",
        "link": "/players"
    },
    {
        "date": "2026-10-30",
        "tag": "update",
        "text": "Power Rankings now grade the team you have today: a mid-season trade for a big scorer moves your ranking right away. Trade value and draft picks no longer move the in-season ranking.",
        "link": "/standings"
    },
    {
        "date": "2026-10-30",
        "tag": "new",
        "text": "Player Rankings quick look: on desktop, hold your cursor on a player to see his 1QB and Superflex values, 30-day trend, last 3 game logs, and next matchup with the defense's Schedule Rankings ease rank, right at the row.",
        "link": "/players"
    },
    {
        "date": "2026-10-30",
        "tag": "fix",
        "text": "Player Rankings sorting: clicking the Value column header now actually sorts by value, and the header controls stay put when switching between Dynasty and Redraft.",
        "link": "/players"
    },
    {
        "date": "2026-10-30",
        "tag": "fix",
        "text": "Top Movers: changing the timeframe or the Dynasty and Redraft toggle now updates the board in place instead of reloading the page. Your scroll position stays put.",
        "link": "/top-movers"
    },
    {
        "date": "2026-10-30",
        "tag": "fix",
        "text": "Trade calculator Player Insights: the Movers time range and format toggle now share one line instead of stacking as two rows of chips.",
        "link": "/trade"
    },
    {
        "date": "2026-10-30",
        "tag": "fix",
        "text": "BR Assistant: asking whether you'll win this week no longer claims there's no scheduled matchup while yours is still loading. It waits for the live matchup or says it's still warming up.",
        "link": "/"
    },
    {
        "date": "2026-10-30",
        "tag": "fix",
        "text": "Trade calculator Player Insights: the Movers scope line under the tabs now fits on one line.",
        "link": "/trade"
    },
    {
        "date": "2026-10-30",
        "tag": "update",
        "text": "Player Rankings, reworked: headshots lead every row, rank movement sits beside the rank, age joins position and team under the name, and the Dynasty / Redraft and 1QB / SF toggles sit beside the title.",
        "link": "/players"
    },
    {
        "date": "2026-10-30",
        "tag": "update",
        "text": "PRO pricing page refresh: the plan picker leads with the free trial strip, the All-Pro card stands out, and a new PRO metrics section groups the gated metrics by category. Plans and prices are unchanged.",
        "link": "/pricing"
    },
    {
        "date": "2026-10-30",
        "tag": "update",
        "text": "ScoreZone game picker: each game now reads as one compact versus line with scores inline once it's underway, and an upcoming game's Box score button is a compact pill.",
        "link": "/scorezone"
    },
    {
        "date": "2026-10-30",
        "tag": "new",
        "text": "Pull to refresh in the app: on ScoreZone and the Weekly Hub, pull down from the top to refresh the scores in place. Your week, league and scroll position stay put.",
        "link": "/scorezone"
    },
    {
        "date": "2026-10-30",
        "tag": "update",
        "text": "Matchups and ScoreZone's classic drive view: the live drive bar is now a slim one-line bar inside the matchup row, with the spot in the possessing team's color and dark mode lifting dark team colors so it stays readable.",
        "link": "/"
    },
    {
        "date": "2026-10-30",
        "tag": "feature",
        "text": "ScoreZone game boards get a field view: a perspective field in team colors with the current drive drawn as a trail of play dots, the first-down line, and a live drive summary. New plays animate in as they arrive, and a touchdown flashes the end zone.",
        "link": "/scorezone"
    },
    {
        "date": "2026-10-29",
        "tag": "fix",
        "text": "ScoreZone game board: the possession dot no longer collides with the clock on phones, and team names and the yard line each appear once instead of twice.",
        "link": "/scorezone"
    },
    {
        "date": "2026-10-29",
        "tag": "fix",
        "text": "Trade Analyst: the positional ranks in the analysis now match the ranks on the trade cards above it, in every format.",
        "link": "/trade"
    },
    {
        "date": "2026-10-29",
        "tag": "fix",
        "text": "Usage risers on the home dashboard: player names now get the full row width, with the usage readout and bars below, so long names no longer truncate.",
        "link": "/"
    },
    {
        "date": "2026-10-29",
        "tag": "fix",
        "text": "Player profiles and compare load faster and stop hanging: league trade history is scanned once per profile, and modal sections time out and retry instead of spinning forever.",
        "link": "/"
    },
    {
        "date": "2026-10-29",
        "tag": "fix",
        "text": "Player profiles: the Recent form graph now loads from a lightweight feed built only from the current season, so it appears faster and no longer disappears when the full game log is slow.",
        "link": "/"
    },
    {
        "date": "2026-10-29",
        "tag": "update",
        "text": "Guides: the index is grouped by category, with a category chip, date and read time on every card, and each article opens with its own dates and section jump pills.",
        "link": "/guides"
    },
    {
        "date": "2026-10-29",
        "tag": "fix",
        "text": "Dark mode: the top 3 dynasty rank badges and team snapshot medals no longer glow. They use deepened metallic tones that read clearly on the dark background.",
        "link": "/"
    },
    {
        "date": "2026-10-29",
        "tag": "update",
        "text": "Player search everywhere is smarter: one shared matcher powers every search box, so punctuation, initials, partial names and small typos all find the right player. Try jamarr, jsn or cmc.",
        "link": "/players"
    },
    {
        "date": "2026-10-29",
        "tag": "update",
        "text": "My Leagues: the Portfolio insights sections now sit as four collapsed rows inside one card, each loading only when you first open it, so the page arrives faster.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-29",
        "tag": "fix",
        "text": "Player cards: the Breakout tab no longer goes missing on a first, cold open, and opening a player from the Breakout Engine lands on the Breakout tab like it should.",
        "link": "/breakouts"
    },
    {
        "date": "2026-10-29",
        "tag": "update",
        "text": "League Health is redesigned around one health card: your score, its verdict and the four things driving it sit together at the top, with history as two compact trend lines.",
        "link": "/league_health"
    },
    {
        "date": "2026-10-29",
        "tag": "update",
        "text": "Under the hood: the app now versions its offline cache from the actual files it ships, so new updates reach your phone on the next load without waiting on a manual cache bump.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-29",
        "tag": "update",
        "text": "The real trade total is on the page now: the Trade Database, Trade Calculator, Trade Intelligence and home page all show how many real Sleeper trades sit behind the values.",
        "link": "/trade-database"
    },
    {
        "date": "2026-10-29",
        "tag": "fix",
        "text": "My Leagues cards: the record band now puts your streak under your record and your standing bigger on the right, and the score strip leads with a week line naming your opponent.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-29",
        "tag": "update",
        "text": "My Leagues: the Front Office full report is now The Brief, a short prose summary of where your leagues stand, the biggest projected gain on the board, and what locks when.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-29",
        "tag": "update",
        "text": "Teams: the signal badges in the team drawer now match the Suggested Moves badges, and the league rank numbers inside the Value by position bar are white so they read cleanly.",
        "link": "/teams"
    },
    {
        "date": "2026-10-29",
        "tag": "update",
        "text": "My Leagues cards are redesigned: your team name headlines each card, your record sits in a tinted band with standing and streak, your weakest and strongest rooms sit under it, and the whole card opens the league.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-29",
        "tag": "fix",
        "text": "Teams and My Leagues: positional room rankings no longer punish you for rostering depth. Rooms are scored against the full set of slots, so adding a player can no longer drop your rank.",
        "link": "/teams"
    },
    {
        "date": "2026-10-29",
        "tag": "update",
        "text": "Teams: the Value by position bar on each team card now carries that team's league rank at the position inside each segment, so the strongest and weakest rooms read straight off the bar.",
        "link": "/teams"
    },
    {
        "date": "2026-10-29",
        "tag": "update",
        "text": "My Leagues: the Front Office full report is a real plan now, walking every open move in priority order with its reason and timing, plus a timeline of what runs before waivers and what locks at kickoff.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-29",
        "tag": "update",
        "text": "Redraft value history now reaches the rest of the site: Beat the Market, Compare charts, trade calculator deltas and the weekly email digest all follow the format you're viewing.",
        "link": "/compare"
    },
    {
        "date": "2026-10-29",
        "tag": "update",
        "text": "Player cards and Top Movers now speak redraft too: in a player card, the value history chart follows the Dynasty/Redraft toggle, and Top Movers gains the same choice, built from redraft values only.",
        "link": "/top-movers"
    },
    {
        "date": "2026-10-28",
        "tag": "fix",
        "text": "Pages no longer reload on their own when a new version lands in the background. The New version banner still offers it, and the page you're reading stays put until you tap.",
        "link": "/"
    },
    {
        "date": "2026-10-28",
        "tag": "update",
        "text": "Schedule Rankings: Adj Avg is back as a column after Ease on desktop, and pins beside the player column on phones so every row keeps a summary number in view.",
        "link": "/schedule"
    },
    {
        "date": "2026-10-28",
        "tag": "update",
        "text": "Schedule Rankings: the table is week by week again, so a whole remaining schedule reads left to right. On phones the same table swipes sideways with rank and player pinned.",
        "link": "/schedule"
    },
    {
        "date": "2026-10-28",
        "tag": "fix",
        "text": "My Leagues: on phones, Rooting Guide game cards now stack the two teams instead of squeezing them side by side, so every player row gets the full width of the card.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-28",
        "tag": "update",
        "text": "My Leagues: the Front Office board is rebuilt around clearing it. Marking a move Done fades it into a Show completed section with Undo, waiver adds name the player to drop, and dismissals now stick across refreshes.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-28",
        "tag": "fix",
        "text": "My Leagues: the Front Office board no longer gets stuck on 'Could not check N of N leagues yet'. The scan reads playoff odds from the cache, the way it was designed to.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-28",
        "tag": "fix",
        "text": "My Leagues: the Rooting Guide no longer repeats the same projection as a range when your leagues agree on a player, and each player's value now sits in a fixed rail on the name line.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-28",
        "tag": "fix",
        "text": "Weekly Hub: the matchup carousel is narrower so the other panels get more room, Scorers shows the top 5 at every position, and the Lineup tab's This Week view is fixed and cleaner.",
        "link": "/"
    },
    {
        "date": "2026-10-28",
        "tag": "fix",
        "text": "Breakout Engine grading is production-first now: a hit needs a real jump in points per game at a startable level, so a small rise off a near-zero baseline no longer counts.",
        "link": "/breakouts"
    },
    {
        "date": "2026-10-28",
        "tag": "fix",
        "text": "PRO plans: fixed yearly subscriptions being cut off after about a month. Yearly PRO now runs for the full term you paid for.",
        "link": "/pricing"
    },
    {
        "date": "2026-10-28",
        "tag": "update",
        "text": "League Awards is rebuilt around the season in front of you: a Title Race strip tracks the current season, league records are tiles with runner-up context, and the all-time standings add losses, points for and scoring average.",
        "link": "/awards"
    },
    {
        "date": "2026-10-28",
        "tag": "fix",
        "text": "My Leagues: the Rooting Guide and Bye Watch now load from a lightweight data pass, so both cards finish loading every time. The numbers and projections are unchanged.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-28",
        "tag": "update",
        "text": "League Stats: charts now fill their cards, three duplicate charts are removed, team names show in full on scatter charts, and long team names get a wider column on bar charts.",
        "link": "/graphs"
    },
    {
        "date": "2026-10-28",
        "tag": "update",
        "text": "My Leagues Rooting Guide: game cards now sit two per row on wide screens, and the team badge in each game header sizes to fit its abbreviation.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-27",
        "tag": "fix",
        "text": "Trade Calculator: evaluating a trade is faster and steadier. Each edit sends one evaluation, and Playoff Impact waits for the simulation to warm up instead of running the numbers on the spot.",
        "link": "/trade"
    },
    {
        "date": "2026-10-27",
        "tag": "update",
        "text": "Weekly Recap and Standings load faster: the duplicate standings, power rankings and lineup efficiency passes are gone, and both pages show the exact same numbers, just sooner.",
        "link": "/standings"
    },
    {
        "date": "2026-10-27",
        "tag": "update",
        "text": "Speed pass across the site: pages and tools do less repeated work behind the scenes, so the Assistant, Front Office report, Dashboard, League Stats and Draft Room all open faster. Numbers and layouts are unchanged.",
        "link": "/"
    },
    {
        "date": "2026-10-27",
        "tag": "update",
        "text": "Weekly Hub: the live matchup views now refresh with far less repeated work behind them. Nothing on the page changes: same numbers, same panels, just less downloading.",
        "link": "/"
    },
    {
        "date": "2026-10-27",
        "tag": "fix",
        "text": "Waivers and Start/Sit: repeat visits load far faster, the Start/Sit strip in the player popup now builds from a single lineup pass for your whole roster, and the Lineup Lab warms up when you reach for its tab instead of on every page load.",
        "link": "/waivers"
    },
    {
        "date": "2026-10-27",
        "tag": "update",
        "text": "Advanced Metrics and Players load faster: the Key Metrics preset pulls its extra columns in two requests instead of eighteen, and the Players list is cached so it survives restarts. Every number is unchanged.",
        "link": "/advanced-metrics"
    },
    {
        "date": "2026-10-27",
        "tag": "fix",
        "text": "ScoreZone: the My Leagues view refreshes far more efficiently, reusing the same data build as the single-league view, and box scores for finished games reopen instantly.",
        "link": "/scorezone"
    },
    {
        "date": "2026-10-27",
        "tag": "fix",
        "text": "My Leagues: Rooting Guide and Bye Watch are built into the page itself now, so they paint with the page instead of loading in afterward, and they keep checking in the background instead of painting an empty card.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-27",
        "tag": "fix",
        "text": "Weekly menu: removed a duplicate red dot. The Tuesday recap-ready banner no longer adds its own dot on top of the Weekly Recap unread dot.",
        "link": "/recap"
    },
    {
        "date": "2026-10-27",
        "tag": "fix",
        "text": "My Leagues: the phone section buttons now land each section right below the top bars instead of scrolling it underneath them, and the sort control now offers Standings.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-27",
        "tag": "fix",
        "text": "Notifications on your phone: the panel now fits the screen, stopping above the bottom nav while the list scrolls inside it, and the red alert dot on the More tab sits on the More icon again.",
        "link": "/"
    },
    {
        "date": "2026-10-27",
        "tag": "fix",
        "text": "Weekly Recap: Week at a Glance now shows the manager username under the High Scorer and Low Scorer team names, instead of repeating the team name.",
        "link": "/recap"
    },
    {
        "date": "2026-10-27",
        "tag": "update",
        "text": "Weekly Hub: the matchup page is rebuilt around your game. Week Leaders run across the top, your matchup stays in view, and Scout is now a gameplan: the verdict, the calls that decide it, and their soft spots.",
        "link": "/"
    },
    {
        "date": "2026-10-27",
        "tag": "update",
        "text": "League Stats: the chart grid fits two columns only when each chart has room, and new charts join the page: Floor / Ceiling, Luck Delta, Scoring Heatmap, Form: Last 3 Weeks and Career Wins by Team.",
        "link": "/graphs"
    },
    {
        "date": "2026-10-27",
        "tag": "update",
        "text": "My Leagues loads faster: the Rooting Guide and Bye Watch answer from a cached result and refresh in the background, and Player Holdings is now sortable by any column.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-27",
        "tag": "update",
        "text": "My Leagues: the Rooting Guide and Bye Watch are cleaned up. Rows are tighter, a player you start in several leagues shows once with all his league badges, and Bye Watch lists the leagues with a hole to fill first.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-27",
        "tag": "fix",
        "text": "Front Office: the moves board no longer sits at Scanned 0 for accounts with many leagues. Every finished league counts right away and the board fills in over a few seconds.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-27",
        "tag": "fix",
        "text": "Trade Analyzer: the navy fair-range bar in the middle of the trade summary track is back. A site-wide tooltip update had quietly knocked it out of place.",
        "link": "/trade"
    },
    {
        "date": "2026-10-27",
        "tag": "fix",
        "text": "BR Assistant now follows the league you are actually in: switching leagues in the app switches the assistant too, instead of it answering for an earlier league.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-27",
        "tag": "fix",
        "text": "Weekly Recap alert dot: opening the recap now clears its dot right away, and the Tuesday recap-ready notification opens the recap itself instead of the matchups page.",
        "link": "/recap"
    },
    {
        "date": "2026-10-27",
        "tag": "update",
        "text": "Weekly Recap: the Injury Report is cleaned up so player names are never cut off, and every team and player name on the page now opens its team or player card.",
        "link": "/recap"
    },
    {
        "date": "2026-10-27",
        "tag": "fix",
        "text": "My Leagues: the Rooting Guide and Bye Watch populate again. Bye Watch reads each starter's actual bye week, and the Rooting Guide keeps retrying while a slow first load lands.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-27",
        "tag": "new",
        "text": "Weekly Recap and Breakout Engine now carry a small red dot in the navigation when there's a new one you haven't opened yet. It clears when you open it.",
        "link": "/recap"
    },
    {
        "date": "2026-10-27",
        "tag": "fix",
        "text": "Team cards and My Leagues cards: playoff odds now read at one decimal and only show 100% or 0% when a team has mathematically clinched a playoff spot or been eliminated. The Teams page drawer shows playoff odds for every team too.",
        "link": "/teams"
    },
    {
        "date": "2026-10-27",
        "tag": "new",
        "text": "Notification center: the bell now opens your alerts, kept newest first with links straight to the league, next to the latest site updates. Settings save to your account, so they apply on every device.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-27",
        "tag": "fix",
        "text": "My Leagues: a league card no longer jumps ahead to next week's preview while its final result is still due. The finished matchup stays on the card.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-27",
        "tag": "fix",
        "text": "Site reliability: pages and My Leagues load more consistently, and Refresh Data now updates each league as it lands instead of blocking on a full rebuild.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-27",
        "tag": "update",
        "text": "Player card Advanced tab: the Trending section is now one verdict instead of a stack of separate alarms, with a Role check line showing your target, share and snap numbers even when they haven't moved.",
        "link": "/metrics"
    },
    {
        "date": "2026-10-26",
        "tag": "fix",
        "text": "Advanced Metrics graphs: zooming now zooms the axes instead of the whole picture, so you can spread out a crowded cluster and read exactly where everyone sits. Drag to pan, Reset to return.",
        "link": "/metrics"
    },
    {
        "date": "2026-10-26",
        "tag": "update",
        "text": "Player game log: the ScoreZone tab in a player's card is now a timeline, with stat tiles up top, a chart of how the points built up play by play, and every play in game order with a running total.",
        "link": "/scorezone"
    },
    {
        "date": "2026-10-26",
        "tag": "update",
        "text": "Schedule Rankings: the page is now a straight rankings table. Compare every team's remaining schedule in the League view, or switch to My Team to see your players in the same order.",
        "link": "/schedule"
    },
    {
        "date": "2026-10-26",
        "tag": "update",
        "text": "My Leagues: the tag on each league card now matches your Teams page. Dynasty leagues show their competitive window, and redraft leagues show Contend, Bubble or Long Shot.",
        "link": "/portfolio"
    },
    {
        "date": "2026-10-25",
        "tag": "fix",
        "text": "ScoreZone: defensive touchdowns and interceptions now credit your DST, live plays keep flowing if the play collector stalls, and the TD alerts bar can be dismissed for the day, with undo.",
        "link": "/scorezone"
    },
    {
        "date": "2026-10-24",
        "tag": "new",
        "text": "Teams: competitive windows rebuilt. A new Powerhouse tier sits above Contender, Holding Pattern splits into Crossroads and Treadmill, and team cards can carry a roster shape tag like Deep, Top-Heavy or Pick-Loaded.",
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
