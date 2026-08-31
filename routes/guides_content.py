"""Original long-form dynasty strategy guides.

This is the crawlable publisher-content layer AdSense reviewers (and search
engines) use to judge the site. Each entry is unique editorial copy, not a
thin wrapper around a tool. Keep bodies in HTML that matches the static-page
style used by ``routes/public_bp.py``.
"""
from __future__ import annotations

GUIDE_AUTHOR_NAME = "hoodiekj"
GUIDE_AUTHOR_URL = "https://youtube.com/@hoodiekj"
GUIDE_PUBLISHED = "2026-06-15"
GUIDE_UPDATED = "2026-08-31"

GUIDES = {
    "dynasty-trade-value": {
        "title": "How Dynasty Trade Value Works",
        "summary": "What a dynasty trade value actually measures, why it differs from "
                   "redraft rankings, and how to read the numbers behind a deal.",
        "published": GUIDE_PUBLISHED,
        "updated": GUIDE_UPDATED,
        "body": """
            <p>
              Every player in a dynasty league carries a <strong>trade value</strong>: a
              single number meant to capture how much that player is worth in the open market
              of league-to-league trades. It is not the same thing as a redraft ranking. A
              redraft ranking answers &ldquo;who scores the most points this season?&rdquo; A
              dynasty value answers &ldquo;what would the rest of the league actually give up to
              acquire this player, accounting for age, contract of expected production, and
              long-term outlook?&rdquo;
            </p>
            <p>
              That distinction is why a 23-year-old breakout receiver can out-value a 30-year-old
              running back who scores more points <em>right now</em>. Dynasty rosters are held for
              years, so the market prices in the runway a player has left, not just this week's
              box score.
            </p>
            <div class="static-section-title">What goes into a value</div>
            <p>
              A good dynasty value blends several inputs rather than relying on any single source:
            </p>
            <ul style="margin-left:20px; line-height:1.8;">
              <li><strong>Consensus market data</strong>: where the crowd of dynasty managers
                  is actually pricing a player. Startup ADP, recent trade results, and public
                  rankings all describe a market, but none of them is the market by itself.</li>
              <li><strong>Recent on-field production</strong>: usage, efficiency, and role,
                  which move a player's stock week to week. A three-game spike in yards is less
                  informative than a three-game spike in snaps and targets.</li>
              <li><strong>Age and position curve</strong>: running backs decline early, wide
                  receivers and quarterbacks hold value far longer. A 27-year-old WR and a
                  27-year-old RB are not in the same phase of their careers.</li>
              <li><strong>Situation</strong>: target share, depth-chart competition, and team
                  context that affect future opportunity. A talented player in a crowded room
                  is priced differently from the same talent with a clean path to snaps.</li>
            </ul>
            <p>
              BR Fantasy recalculates these inputs daily and calibrates the result against
              observed dynasty prices, with guardrails so one noisy week cannot send a veteran
              from a third-round startup pick to a first. You can see calibrated values for
              every relevant player on the
              <a href="/rankings/dynasty">dynasty rankings</a> page, or browse the full
              <a href="/dynasty-trade-value-chart">dynasty trade value chart</a> to compare across
              positions at a glance.
            </p>
            <div class="static-section-title">Why two values for the same player?</div>
            <p>
              Most dynasty leagues are either single-quarterback (1QB) or Superflex, and a player's
              value can change dramatically between the two formats. Quarterbacks are far more
              valuable in Superflex because you can start two of them. If your league is Superflex,
              always look at Superflex values, using 1QB numbers will badly under-rate every
              passer. We cover this in depth in
              <a href="/guides/superflex-vs-1qb">Superflex vs 1QB</a>.
            </p>
            <p>
              Scoring also moves the number. Tight-end premium, full PPR, and bonus points for
              long scores all change which archetypes the market pays up for. A value is only
              useful if it was built for a league that looks like yours. If your league pays
              extra for tight ends, start with
              <a href="/guides/te-premium-leagues">TE premium strategy</a> before you treat a
              1-PPR chart as gospel.
            </p>
            <div class="static-section-title">How to read a number on the chart</div>
            <p>
              A value is a <em>relative</em> price, not a prediction of next week's points.
              Compare players to each other, not to a fantasy of what the number &ldquo;should&rdquo;
              be. A 42 and a 28 is a meaningful gap; a 42.1 and a 41.8 is noise. When two players
              sit in the same band, the decision is about fit: age, position need, and whether
              you are
              <a href="/guides/contending-in-dynasty">contending</a> or
              <a href="/guides/dynasty-rebuild-strategy">rebuilding</a>.
            </p>
            <p>
              Watch movement, not just the snapshot. The
              <a href="/top-movers">top movers</a> page is where a value becomes a trading
              window: a player whose number is falling on usage, not just on a cold box score,
              is a different kind of dip than a player who scored 4 points on 90% of snaps.
              Pair the chart with
              <a href="/guides/reading-advanced-metrics">advanced metrics</a> before you decide
              the market is wrong.
            </p>
            <div class="static-section-title">What a value is not</div>
            <p>
              It is not a projection, a start/sit grade, or a guarantee that a trade will
              &ldquo;win.&rdquo; It does not know that you already have three receivers and zero
              running backs. It does not know that your league mates refuse to trade first-round
              picks in-season. Those constraints are why the
              <a href="/trade">trade calculator</a> exists: it applies the same values to both
              sides of a deal and then leaves room for you to judge fit. For a step-by-step
              process, see
              <a href="/guides/evaluating-a-trade">how to evaluate a dynasty trade</a>.
            </p>
            <div class="highlight-box">
              Bottom line: a dynasty value is a market estimate, not a law. Use it as the starting
              point for a negotiation, then adjust for your roster's timeline and needs.
            </div>
        """,
    },
    "superflex-vs-1qb": {
        "title": "Superflex vs 1QB: Why the Same Player Has Two Values",
        "summary": "Quarterbacks dominate Superflex leagues. Here's how values shift between "
                   "formats and how to avoid badly mispricing a trade.",
        "published": GUIDE_PUBLISHED,
        "updated": GUIDE_UPDATED,
        "body": """
            <p>
              The single biggest factor in a player's dynasty value, bigger than age,
              bigger than last week's stat line, is often just your league format. In a
              <strong>single-quarterback (1QB)</strong> league you start one QB. In a
              <strong>Superflex</strong> league you can start a second quarterback in a flex spot,
              which makes the position enormously more valuable.
            </p>
            <div class="static-section-title">Why quarterbacks explode in Superflex</div>
            <p>
              There are only 32 starting NFL quarterbacks, and in a 12-team Superflex league up to
              24 of them can be in starting lineups every week. That scarcity means even mid-tier
              starters carry real weight, and the elite young passers become the most valuable
              assets in the entire player pool, frequently worth more than any running back
              or receiver.
            </p>
            <p>
              In 1QB, the opposite is true: you only need one quarterback, streamable options are
              everywhere, and so the position is heavily discounted. Top-tier wide receivers and
              running backs sit at the top of 1QB value charts instead.
            </p>
            <p>
              The math is simple scarcity. If 12 teams each start one quarterback, the 13th-best
              passer is a backup. If 12 teams each start two, the 20th-best passer is still a
              weekly starter. That second cohort, the Kirk Cousins / Baker Mayfield band in a
              given year, is where Superflex leagues are won and 1QB leagues barely notice.
            </p>
            <div class="static-section-title">The practical trap</div>
            <p>
              The most common dynasty trade mistake is using the wrong format's values. If you play
              Superflex but evaluate a quarterback trade with 1QB numbers, you will think you are
              winning a deal while actually giving up a premium asset for pennies. Always confirm
              which format a value reflects before you commit.
            </p>
            <p>
              On the <a href="/rankings/dynasty">dynasty rankings</a> you can view values for the
              format your league uses, and the
              <a href="/trade">trade calculator</a> lets you toggle Superflex so both sides of a
              deal are priced correctly.
            </p>
            <div class="static-section-title">How the rest of the board moves</div>
            <p>
              Superflex does not only inflate quarterbacks. It compresses everyone else. Elite
              running backs and receivers still matter, but they occupy a smaller share of total
              league value because so much of the pie is locked in the passer market. In
              practical terms:
            </p>
            <ul style="margin-left:20px; line-height:1.8;">
              <li>A top-five Superflex quarterback often costs what a top-three 1QB skill player
                  costs. Do not &ldquo;feel&rdquo; that as a rip-off; it is the format working.</li>
              <li>Mid-round startup running backs are relatively cheaper in Superflex, because
                  managers spent early capital on passers. That is a feature if you are
                  <a href="/guides/startup-draft-guide">building a startup</a> around depth.</li>
              <li>Rookie firsts still matter, but a late first that projects as a quarterback
                  with a path to starting is Superflex gold and 1QB noise. See
                  <a href="/guides/rookie-draft-strategy">rookie draft strategy</a>.</li>
            </ul>
            <div class="static-section-title">Roster construction differences</div>
            <p>
              In Superflex, quarterback depth is not a luxury, it is insurance. One injury can
              turn a contender into a team starting a backup in the flex. Holding three startable
              passers is a common, rational strategy; holding five is usually dead capital unless
              you plan to sell into a QB-starved league. In 1QB, a second quarterback is a
              handcuff, not a cornerstone.
            </p>
            <p>
              When you are offered a &ldquo;fair&rdquo; deal that sends your QB2 for a young
              receiver, ask what your lineup looks like in week 14 if your QB1 misses time.
              The
              <a href="/guides/evaluating-a-trade">trade evaluation process</a> exists for
              exactly this kind of format-aware check, not just adding the numbers.
            </p>
            <div class="highlight-box">
              Rule of thumb: in Superflex, treat startable quarterbacks as premium assets. In 1QB,
              let the other manager overpay for them.
            </div>
        """,
    },
    "reading-advanced-metrics": {
        "title": "Reading Advanced Metrics: A Fantasy Manager's Guide",
        "summary": "Target share, air yards, snap counts, red-zone usage and more, what "
                   "each metric tells you and which ones actually predict fantasy points.",
        "published": GUIDE_PUBLISHED,
        "updated": GUIDE_UPDATED,
        "body": """
            <p>
              Box-score stats tell you what already happened. <strong>Advanced metrics</strong>
              tell you whether it is likely to keep happening. They separate players who are
              producing because of genuine, repeatable opportunity from those riding unsustainable
              efficiency or touchdown luck. Here's how to read the ones that matter.
            </p>
            <div class="static-section-title">Opportunity metrics (the most predictive)</div>
            <ul style="margin-left:20px; line-height:1.8;">
              <li><strong>Target share</strong>: the percentage of his team's targets a
                  receiver or tight end earns. A rising target share is one of the strongest
                  leading indicators of future fantasy production.</li>
              <li><strong>Snap share</strong>: how often a player is actually on the field.
                  Low snap share caps a player's ceiling no matter how efficient he looks.</li>
              <li><strong>Air yards</strong>: the total downfield distance of a player's
                  targets. High air yards signal a player is being used in a high-value role even
                  before the catches show up.</li>
              <li><strong>Red-zone usage</strong>: touches and targets inside the 20.
                  Red-zone volume drives touchdowns, which are the most volatile (and valuable)
                  source of fantasy points.</li>
            </ul>
            <div class="static-section-title">Efficiency metrics (context, not gospel)</div>
            <p>
              Yards per route run, yards after catch, and yards per touch describe how well a
              player converts opportunity into production. They are useful, but efficiency is far
              noisier than volume, a great yards-per-touch number on five touches a game
              won't survive a larger sample. Always weigh efficiency against the opportunity behind
              it.
            </p>
            <p>
              A practical filter: if a running back is 95th percentile in yards per carry and
              20th percentile in snap share, you are looking at a change-of-pace back having a
              hot month, not a league-winner. If a receiver is 40th percentile in yards per
              route and 90th percentile in target share, you are looking at a volume earner
              whose fantasy floor is safer than the efficiency scouts admit.
            </p>
            <div class="static-section-title">How to use them together</div>
            <p>
              The players worth buying are the ones whose opportunity is climbing before the
              fantasy points catch up: rising snaps, rising target share, growing red-zone role.
              That gap between opportunity and output is exactly what the
              <a href="/breakouts">breakout engine</a> is built to surface, and you can dig into
              the underlying numbers on the <a href="/players">player database</a>.
            </p>
            <p>
              The opposite gap is a sell signal. A player whose points are running ahead of
              snaps, targets, and red-zone looks is the classic
              <a href="/guides/buy-low-sell-high">sell-high</a> candidate. Touchdowns cluster.
              Markets overfit to the last four games. Metrics keep you honest when the box
              score is screaming.
            </p>
            <div class="static-section-title">Sample size and when to trust a trend</div>
            <p>
              One week of 22% target share is a headline. Six weeks of 22% target share is a
              role. Early-season usage is noisy because coaching staffs are still sorting
              personnel; by week six, snap and target shares have usually settled enough to
              trade on. Injuries and depth-chart changes reset the clock: a new starter's
              first two games are more informative than a veteran' s quiet week in a blowout.
            </p>
            <p>
              Age still sits underneath every metric. A 24-year-old with rising snaps is a
              different bet from a 29-year-old with the same chart, because
              <a href="/guides/positional-aging-curves">positional aging curves</a> say the
              younger player can still grow into the role. Metrics describe the present;
              dynasty value has to price the future.
            </p>
            <div class="highlight-box">
              Prioritize volume and role over efficiency. Opportunity is sticky; efficiency
              regresses.
            </div>
        """,
    },
    "rookie-draft-strategy": {
        "title": "Dynasty Rookie Draft Strategy",
        "summary": "How to value rookie picks, read prospect profiles, and avoid the most common "
                   "first-year-player mistakes in dynasty.",
        "published": GUIDE_PUBLISHED,
        "updated": GUIDE_UPDATED,
        "body": """
            <p>
              The rookie draft is where dynasty championships are quietly built. Cheap, ascending
              young talent is the best value in the format, but rookie picks are also where
              managers most often overpay for hype. Here's a framework for drafting well.
            </p>
            <div class="static-section-title">Value the picks, then the players</div>
            <p>
              Before you fall in love with a prospect, understand what the pick itself is worth.
              Early first-round rookie picks carry significant trade value because of their upside,
              but that value drops quickly as you move into the second and third rounds. Knowing the
              market price of a pick keeps you from trading a proven player for a lottery ticket.
            </p>
            <p>
              A useful habit: price the pick on your
              <a href="/dynasty-trade-value-chart">trade value chart</a> the way you would price
              a player. If a late first is worth roughly a mid-tier veteran, do not spend a
              league-winning starter to &ldquo;get younger&rdquo; unless the prospect's median
              outcome actually beats that veteran on your timeline.
            </p>
            <div class="static-section-title">What actually predicts rookie success</div>
            <ul style="margin-left:20px; line-height:1.8;">
              <li><strong>Draft capital</strong>: where the NFL drafted a player is one of
                  the best predictors of opportunity. Teams invest snaps and targets in the players
                  they spent premium picks on.</li>
              <li><strong>Landing spot</strong>: the same prospect can be a league-winner or
                  a redraft afterthought depending on depth-chart competition and offensive
                  quality.</li>
              <li><strong>College production at a young age</strong>: players who dominated
                  early in their college careers (a strong &ldquo;breakout age&rdquo;) hit at higher
                  rates.</li>
              <li><strong>Athletic profile</strong>: testing scores like RAS provide a floor
                  check, especially at receiver and running back.</li>
            </ul>
            <div class="static-section-title">Position priorities</div>
            <p>
              In most formats, prioritize wide receivers early, they have the longest dynasty
              shelf life and the highest hit rate near the top of rookie drafts. Running backs offer
              immediate production but age out fast, so they are better targeted by contending teams.
              In Superflex, a rookie quarterback with a clear path to starting can be worth a top
              pick on its own.
            </p>
            <p>
              Tight ends are a patience position. Even good prospects often take two years to
              become weekly starters. That is fine in a rebuild and painful on a contender.
              If your league uses TE premium, the calculus changes; see
              <a href="/guides/te-premium-leagues">TE premium leagues</a>.
            </p>
            <p>
              You can study full prospect profiles, college metrics, draft capital, athletic
              scores, and live ADP movement, on the <a href="/prospects">rookie prospects</a>
              page.
            </p>
            <div class="static-section-title">Common rookie-draft mistakes</div>
            <ul style="margin-left:20px; line-height:1.8;">
              <li><strong>Drafting the jersey, not the role.</strong> Name recognition from
                  college broadcasts is not a projection. Check the depth chart.</li>
              <li><strong>Reaching for running backs in a rebuild.</strong> You will own a
                  26-year-old committee back when your window actually opens.</li>
              <li><strong>Ignoring Superflex QB paths.</strong> A second-round passer who
                  can start in year two is often the best pick on the board and the one your
                  league-mates skip for a flashy receiver.</li>
              <li><strong>Trading future firsts in a panic.</strong> Next year's first is
                  an option on a player who does not exist yet. Selling it to patch a one-week
                  hole is how contenders quietly become average.</li>
            </ul>
            <div class="highlight-box">
              Draft talent and opportunity, not name recognition. The best rookie picks are the
              ones your league mates aren't talking about yet.
            </div>
        """,
    },
    "buy-low-sell-high": {
        "title": "Buy-Low and Sell-High: Timing the Dynasty Market",
        "summary": "Dynasty value is always moving. Learn to recognize the windows where you can "
                   "buy a player below his real worth or sell above it.",
        "published": GUIDE_PUBLISHED,
        "updated": GUIDE_UPDATED,
        "body": """
            <p>
              Dynasty trade value is not static, it moves constantly with injuries, depth
              chart changes, hot streaks, and slumps. The managers who win their leagues over time
              are the ones who trade <em>against</em> these short-term swings: buying players the
              market has soured on and selling players it has temporarily overrated.
            </p>
            <div class="static-section-title">When to buy low</div>
            <ul style="margin-left:20px; line-height:1.8;">
              <li>A talented player in a brief slump whose underlying usage (snaps, target share)
                  is still strong.</li>
              <li>A young player stuck behind an aging or injury-prone starter who will eventually
                  get the job.</li>
              <li>A player coming off a minor injury, where the panic is bigger than the long-term
                  risk.</li>
            </ul>
            <p>
              The buy-low that actually works is boring. You are not hunting a player who
              &ldquo;looked bad.&rdquo; You are hunting a player whose <em>role</em> is intact
              while his <em>results</em> are ugly. That is why
              <a href="/guides/reading-advanced-metrics">opportunity metrics</a> matter more
              than a two-game point drought. If snaps and targets are still there, the points
              usually return. If snaps are gone, the drought is the news.
            </p>
            <div class="static-section-title">When to sell high</div>
            <ul style="margin-left:20px; line-height:1.8;">
              <li>A player riding an unsustainable touchdown rate that his opportunity won't
                  support.</li>
              <li>An aging running back coming off a big stretch, sell the name before the
                  cliff.</li>
              <li>A backup who spiked in value during a short injury fill-in for a starter who is
                  about to return.</li>
            </ul>
            <p>
              Selling high is socially harder than buying low. Your league-mates just watched
              the player eat, and they do not want to pay peak price. That is fine. You do not
              need to extract the absolute top. You need to move a declining or lucky profile
              into a younger or stickier one before the market catches up. A good sell is one
              you still like a little; if you are desperate to dump the player, you waited too
              long.
            </p>
            <div class="static-section-title">Let the data find the windows</div>
            <p>
              The clearest buy-low and sell-high signals show up as movement in value over time.
              The <a href="/top-movers">top movers</a> page tracks which players are rising and
              falling fastest, and <a href="/trade-intel">trade intelligence</a> surfaces market
              signals from real league activity. Pair those with the
              <a href="/rankings/dynasty">current rankings</a> to spot gaps between a player's price
              and his true outlook.
            </p>
            <p>
              Then check the deal against your timeline. A buy-low running back is a gift to a
              <a href="/guides/contending-in-dynasty">contender</a> and a trap to a
              <a href="/guides/dynasty-rebuild-strategy">rebuilder</a> who just added another
              27-year-old. The same dip can be a win or a mistake depending on whether you
              need 2026 points or 2028 optionality.
            </p>
            <div class="static-section-title">Do not manufacture a take</div>
            <p>
              Not every quiet week is a buy, and not every spike is a sell. Some players are
              simply good, and the market is correctly expensive. Forced contrarianism is how
              managers accumulate a roster of &ldquo;process wins&rdquo; that never start. If
              the metrics, the age curve, and the depth chart all agree with the price, leave
              the player alone and look at the next name on the movers list.
            </p>
            <div class="highlight-box">
              The market overreacts to recent results. Your edge is patience: buy the dip on talent,
              sell the spike on age and luck.
            </div>
        """,
    },
    "evaluating-a-trade": {
        "title": "How to Evaluate a Dynasty Trade",
        "summary": "A step-by-step process for judging any trade offer, beyond just adding "
                   "up the values on each side.",
        "published": GUIDE_PUBLISHED,
        "updated": GUIDE_UPDATED,
        "body": """
            <p>
              Adding up trade values on each side of a deal is a useful first check, but it is only
              the beginning. The best trades aren't always the ones that &ldquo;win&rdquo; on raw
              value, they're the ones that make <em>your</em> roster better for <em>your</em>
              timeline. Here's a repeatable process.
            </p>
            <div class="static-section-title">Step 1: Check the raw value</div>
            <p>
              Start by comparing the total value on each side using format-appropriate numbers
              (1QB or Superflex). A quick way to do this is the
              <a href="/trade">trade calculator</a>, which grades both sides and suggests counters.
              If a deal is wildly lopsided on value, you usually have your answer.
            </p>
            <p>
              Use the same settings your league actually plays. Superflex vs 1QB is the big
              one (see <a href="/guides/superflex-vs-1qb">Superflex vs 1QB</a>), but TE premium
              and roster size also move the result. A deal that looks fair on a generic chart
              can be a steal or a disaster once those knobs match your league.
            </p>
            <div class="static-section-title">Step 2: Account for consolidation</div>
            <p>
              Two good players are generally worth more than three mediocre ones, because starting
              lineup spots are limited and the best players are the hardest to replace. When you
              trade multiple pieces for one stud, expect, and accept, paying a small
              value premium for that consolidation.
            </p>
            <p>
              The reverse is true in a rebuild: unpacking a star into several younger pieces
              can be correct even if you &ldquo;lose&rdquo; a few points of value, because you
              cannot start the star enough times to justify holding him through a two-year
              trough. That is a timeline decision, not a calculator error.
            </p>
            <div class="static-section-title">Step 3: Match the deal to your timeline</div>
            <p>
              Are you contending or rebuilding? Contenders should trade youth and picks for proven,
              win-now production. Rebuilders should do the reverse: sell aging stars for young
              players and draft capital. A trade that's &ldquo;fair&rdquo; on value can still be
              wrong if it doesn't fit where your team is in its cycle.
            </p>
            <p>
              If you are unsure which mode you are in, count startable weeks and upcoming
              draft capital, not last year's record. A 5-9 team with three firsts is a
              rebuild even if the chat still thinks you are &ldquo;a quarterback away.&rdquo;
              Walk through
              <a href="/guides/dynasty-rebuild-strategy">rebuild strategy</a> or
              <a href="/guides/contending-in-dynasty">contending strategy</a> before you
              accept a deal that fights your window.
            </p>
            <div class="static-section-title">Step 4: Value positional scarcity and need</div>
            <p>
              A player is worth more to a roster that needs his position. Don't trade from a
              position of strength into another position of strength, address real lineup
              holes. In Superflex, weigh quarterback depth especially heavily.
            </p>
            <div class="static-section-title">Step 5: Look past this week</div>
            <p>
              Before you finalize, sanity-check the underlying trends from the
              <a href="/guides/reading-advanced-metrics">advanced metrics</a> and the
              <a href="/top-movers">top movers</a> page. You want to be buying ascending players and
              selling declining ones, not the reverse.
            </p>
            <p>
              Then sleep on any deal that moves a cornerstone. Dynasty trades are rarely so
              urgent that you must accept before the next snap. If the other manager is
              rushing you, that is information too.
            </p>
            <div class="highlight-box">
              A good trade makes your starting lineup better for your timeline. Value is the
              starting point; fit is the decision.
            </div>
        """,
    },
    "dynasty-rebuild-strategy": {
        "title": "How to Rebuild a Dynasty Roster",
        "summary": "When to tear it down, which assets to sell, and how to restock with youth "
                   "and picks without wasting two seasons.",
        "published": GUIDE_UPDATED,
        "updated": GUIDE_UPDATED,
        "body": """
            <p>
              A rebuild is not a vibe. It is a decision that your current roster cannot
              reasonably win the league this season or next, so you will trade present
              production for future optionality. Done well, a rebuild lasts one offseason and
              one ugly year. Done poorly, it becomes a five-year hobby of collecting
              &ldquo;upside&rdquo; that never starts.
            </p>
            <div class="static-section-title">Admit the window is closed</div>
            <p>
              Look at age, picks, and starting lineup quality together. If your skill-position
              core is 27-plus, you have no first-round picks in the next two drafts, and you
              are already out of the playoff picture by week eight, you are not &ldquo;a
              running back away.&rdquo; You are a seller. Staying in the middle, good enough
              to finish 7-7, is the most expensive place in dynasty.
            </p>
            <p>
              Public tools help you be honest. Rank your roster against the
              <a href="/rankings/dynasty">dynasty rankings</a>, then ask whether your starters
              would still be starters in two years. If the answer is no, start the rebuild
              before the rest of the league notices.
            </p>
            <div class="static-section-title">Sell the right aging pieces</div>
            <p>
              Your best trade chips are productive veterans that contenders need this season:
              aging running backs, proven receivers still posting target share, and (in
              Superflex) a quarterback you cannot wait on. Price them with
              <a href="/guides/dynasty-trade-value">dynasty trade values</a>, then prefer
              packages that return young players with roles plus draft capital, not just a
              pile of thirds.
            </p>
            <p>
              Do not sell young players with sticky usage just because you are rebuilding.
              A 23-year-old WR2 with 20% target share <em>is</em> the rebuild. Selling him
              for a future first feels active and is often a downgrade.
            </p>
            <div class="static-section-title">What you should be collecting</div>
            <ul style="margin-left:20px; line-height:1.8;">
              <li><strong>Rookie firsts and early seconds</strong>, especially in years with
                  a known quarterback or receiver class. See
                  <a href="/guides/rookie-draft-strategy">rookie draft strategy</a>.</li>
              <li><strong>Young players whose opportunity is rising</strong> before the
                  points show up, the same profiles the
                  <a href="/breakouts">breakout engine</a> is built to flag.</li>
              <li><strong>Quarterback youth in Superflex</strong>, even if they are sitting
                  this year. Rebuilds that ignore passers in Superflex restart in three
                  years.</li>
            </ul>
            <div class="static-section-title">How long it should take</div>
            <p>
              If you still cannot name a future starting lineup after two rookie drafts, you
              either sold the wrong players or you kept drafting running backs. A healthy
              rebuild produces a competitive roster as soon as the young core's usage
              arrives, not when every pick has &ldquo;hit.&rdquo; Switch from collecting to
              <a href="/guides/contending-in-dynasty">contending</a> the moment your starting
              lineup can actually win weeks. Holding picks past that point is how rebuilders
              miss their window on the way up.
            </p>
            <div class="static-section-title">Trades you should not take just to &ldquo;get younger&rdquo;</div>
            <p>
              Youth is not automatically good. A 22-year-old with 8% snap share and a
              crowded depth chart is not a building block; he is a dart. A 26-year-old
              receiver with 22% target share may be the best remaining core piece you
              have. Rebuilds fail when every productive veteran is sold for a pick that
              will be used on a running back two years from now.
            </p>
            <p>
              Use the <a href="/trade">trade calculator</a> to keep yourself honest, then
              apply the same
              <a href="/guides/evaluating-a-trade">fit checks</a> you would in any other
              deal. If the package does not leave you with startable youth or premium
              picks, you did not rebuild. You just got worse.
            </p>
            <div class="highlight-box">
              Tear down once, sell aging production for youth and picks, and stop rebuilding
              the week your young core can win games.
            </div>
        """,
    },
    "contending-in-dynasty": {
        "title": "How to Contend in Dynasty Leagues",
        "summary": "How to recognize a real championship window and which trades actually "
                   "push a good team over the top.",
        "published": GUIDE_UPDATED,
        "updated": GUIDE_UPDATED,
        "body": """
            <p>
              Contending is the opposite of collecting. You already have a starting lineup that
              can win a title this season, so every trade should be judged by whether it
              improves the weeks you will actually play, especially the playoff slate, not by
              whether it makes your roster &ldquo;younger&rdquo; on paper.
            </p>
            <div class="static-section-title">Confirm you are actually a contender</div>
            <p>
              Record is a lagging indicator. Look at starting-lineup value, remaining schedule,
              and whether your core still has a year of production left. A 6-3 team of 29-year-old
              running backs can be a contender this year and a rebuild in March. A 4-5 team with
              elite young skill players and a soft playoff schedule can still be a buyer.
            </p>
            <p>
              Use the <a href="/rankings/dynasty">rankings</a> and your league's
              playoff-odds tools to sanity-check the chat. If you are a true bubble team,
              small buys (a RB2, a QB2 in Superflex) beat blockbuster sells of future firsts.
            </p>
            <div class="static-section-title">What contenders should buy</div>
            <ul style="margin-left:20px; line-height:1.8;">
              <li><strong>Reliable weekly production</strong> at positions you actually start.
                  Volume running backs and high-floor receivers beat dart-throw rookies in
                  November.</li>
              <li><strong>Injury insurance</strong> at your thinnest position. One hamstring
                  should not turn a title favorite into a streamer.</li>
              <li><strong>Quarterback depth in Superflex.</strong> Format scarcity makes a
                  startable QB2 more valuable in December than another young WR4. See
                  <a href="/guides/superflex-vs-1qb">Superflex vs 1QB</a>.</li>
            </ul>
            <div class="static-section-title">What contenders should not do</div>
            <p>
              Do not trade a cornerstone who is still in his prime to &ldquo;get picks
              back&rdquo; during a year you can win. Do not buy aging running backs in August
              if your window is actually next season. Do not empty the taxi squad of every
              interesting young player just to add a committee back who will be irrelevant
              in 14 months, unless that back is the difference in a title week.
            </p>
            <p>
              The right cost is usually a future first plus a depth piece, or a young player
              who is not in your starting lineup. Price it with the
              <a href="/trade">trade calculator</a> and the process in
              <a href="/guides/evaluating-a-trade">how to evaluate a trade</a>. Paying a
              premium is expected. Paying two future firsts for a rental who does not start
              for you is how contenders accidentally rebuild.
            </p>
            <div class="static-section-title">In-season vs offseason buys</div>
            <p>
              In-season, prioritize players with roles <em>now</em>. Offseason, you can still
              contend and improve the long-term core at the same time, especially at receiver
              and quarterback, where
              <a href="/guides/positional-aging-curves">aging curves</a> are slower. The
              offseason is also when rebuilders overpay for your aging backs. That is the
              <a href="/guides/buy-low-sell-high">sell-high</a> window you should use if a
              veteran just carried you through a title run.
            </p>
            <div class="static-section-title">Lineup weeks, not trophy case</div>
            <p>
              Contending is about the weeks you will start a player, not about collecting
              every name that might be good in 2028. If a young receiver is your WR5, he is
              a trade chip for a back who will start in December. If your Superflex QB2 is
              a streamer, fix that before you add another prospect.
            </p>
            <p>
              Check remaining schedule and injury risk the same way you would in redraft.
              Dynasty value still matters so you do not torch the future for a two-week
              rental, but a title is worth a future late first. It is rarely worth two
              early firsts and your cheapest young starter.
            </p>
            <div class="highlight-box">
              If the lineup can win the league, buy production that starts. Save the youth
              movement for the winter after a real window closes.
            </div>
        """,
    },
    "te-premium-leagues": {
        "title": "TE Premium Dynasty Strategy",
        "summary": "How extra tight-end scoring changes startup drafts, trades, and which "
                   "archetypes are actually worth paying up for.",
        "published": GUIDE_UPDATED,
        "updated": GUIDE_UPDATED,
        "body": """
            <p>
              Tight-end premium (TEP) leagues award extra points per reception, or extra
              points per tight-end reception, on top of whatever PPR the rest of the roster
              uses. The usual bump is +0.5 PPR for tight ends. That sounds small. Over a
              season it is not. It stretches the gap between the few tight ends who earn
              real volume and everyone else, and it changes how you should spend draft
              capital and trade chips.
            </p>
            <div class="static-section-title">Why the elite tier gets expensive</div>
            <p>
              In standard PPR, a tight end competing with receivers for targets is often a
              positional afterthought. In TEP, those same targets are worth more than a
              receiver's. The handful of tight ends who see 20% or more of their team's
              targets become weekly difference-makers, not just &ldquo;TE1s.&rdquo; The
              market responds by pushing them up
              <a href="/rankings/dynasty">dynasty rankings</a> relative to a non-premium chart.
            </p>
            <p>
              That does not mean every tight end is a buy. The position is still bimodal:
              a small premium tier, a wide middle that is startable but not special, and a
              long tail of streaming options. Paying a first-round startup pick for a
              mid-tier tight end because &ldquo;it's TEP&rdquo; is how you strand capital.
            </p>
            <div class="static-section-title">Startup and rookie draft effects</div>
            <p>
              In a TEP startup, it is reasonable to take a true difference-making tight end
              earlier than you would in a 1-PPR league, especially if you are already set
              at quarterback in Superflex. It is not reasonable to take the TE8 over a
              young receiver with a clear role. Use
              <a href="/guides/startup-draft-guide">startup draft strategy</a> and check
              values on the
              <a href="/dynasty-trade-value-chart">trade value chart</a> rather than
              following a generic &ldquo;TEP cheat sheet&rdquo; that ignores the rest of
              your build.
            </p>
            <p>
              In rookie drafts, TEP raises the floor of early-declare tight ends with
              draft capital, but it does not erase development time. Most tight ends still
              take a year or two. Rebuilders can wait; contenders should prefer a proven
              volume tight end over a prospect unless the prospect is a clear premium
              talent. See <a href="/guides/rookie-draft-strategy">rookie draft strategy</a>.
            </p>
            <div class="static-section-title">Trading in TEP</div>
            <p>
              When you evaluate a deal, make sure both sides are scored as TEP. A tight
              end who looks &ldquo;expensive&rdquo; on a standard chart is often fairly
              priced once the premium is applied. The
              <a href="/trade">trade calculator</a> is the place to toggle that, then apply
              the same
              <a href="/guides/evaluating-a-trade">fit checks</a> you would for any other
              position: need, age, and whether you are contending.
            </p>
            <p>
              Holding two premium tight ends is a real strategy in TEP because the waiver
              replacements are so much worse than at receiver. Holding four is usually a
              traffic jam. Trade the third for a need, do not wait for a perfect offer
              that never comes.
            </p>
            <div class="static-section-title">What the extra half-point actually does</div>
            <p>
              Over 16 games, 0.5 extra PPR on 80 catches is 40 points. That is a
              meaningful season total, which is why volume tight ends separate from
              blocking specialists who luck into six touchdowns. When you are deciding
              between a TE with 7 targets a week and a WR with 7 targets a week, TEP
              is the only format where the tight end often wins that comparison.
            </p>
            <p>
              It still does not beat a true WR1's target share. Draft and trade as if
              TEP is a scoring tweak that enlarges one position's premium tier, not a
              new sport. The
              <a href="/rankings/dynasty">rankings</a> already bake that into the
              number if you have the setting on.
            </p>
            <div class="highlight-box">
              TEP makes true volume tight ends worth paying up for. It does not make every
              tight end a first-round asset.
            </div>
        """,
    },
    "startup-draft-guide": {
        "title": "Dynasty Startup Draft Strategy",
        "summary": "How to build a roster from scratch: where to spend early picks, when to "
                   "take quarterbacks, and how to avoid a pretty team that never wins.",
        "published": GUIDE_UPDATED,
        "updated": GUIDE_UPDATED,
        "body": """
            <p>
              A dynasty startup is the one draft where the entire player pool is available
              at once. That makes it the highest-leverage event in the format and the
              easiest place to copy a flashy board instead of building a team. The goal is
              not to draft the most famous names. It is to leave with a starting lineup
              that has a timeline and enough depth to survive the first injury wave.
            </p>
            <div class="static-section-title">Pick a window before pick 1.01</div>
            <p>
              Decide whether this startup is a win-now build, a balanced core, or a youth
              pile. You can adjust later, but the first four rounds should not fight each
              other. Mixing a 30-year-old bell-cow, two rookie receivers, and no
              quarterback plan in Superflex is how startups become immediate rebuilds.
            </p>
            <p>
              If you want to contend early, bias toward proven volume and accept that some
              of those players will need to be sold in two years. If you want to grow,
              bias toward young receivers and (in Superflex) young passers, and be willing
              to be bad in year one. Both are valid.
              <a href="/guides/contending-in-dynasty">Contending</a> and
              <a href="/guides/dynasty-rebuild-strategy">rebuilding</a> are just those
              plans with the draft already over.
            </p>
            <div class="static-section-title">Format first, then position</div>
            <p>
              In Superflex, elite and good-enough quarterbacks come off the board early
              for a reason. Falling behind at the position in round two is a hole you will
              spend three years filling. In 1QB, let someone else take the extra passer
              and spend the capital on skill-position talent. Details are in
              <a href="/guides/superflex-vs-1qb">Superflex vs 1QB</a>.
            </p>
            <p>
              Tight-end premium similarly pulls true volume TEs up the board. Everyone
              else should still be ranked by the same
              <a href="/guides/dynasty-trade-value">trade-value</a> logic you will use
              after the draft: age, role, and replacement cost.
            </p>
            <div class="static-section-title">ADP is a clock, not a ranking</div>
            <p>
              Startup ADP tells you when you will have to reach. It does not tell you who
              is better. If a player you want is going two rounds later than his value,
              wait. If the player you need never makes it back, that is the cost of
              nominating a plan. Compare ADP to the
              <a href="/dynasty-trade-value-chart">value chart</a> and to
              <a href="/guides/adp-vs-trade-value">ADP vs trade value</a> so you are not
              drafting a consensus board that already has no edge.
            </p>
            <div class="static-section-title">Middle-round discipline</div>
            <p>
              Rounds 5 through 12 are where startups are decided. This is where managers
              either stack young receivers with roles or panic-draft a running back
              committee. Prefer players with snaps and targets, even if they are less
              famous. Use
              <a href="/guides/reading-advanced-metrics">advanced metrics</a> and
              <a href="/prospects">prospect profiles</a> instead of recency from last
              year's playoffs.
            </p>
            <p>
              Late, take upside that does not need a starting spot this year: rookies with
              draft capital, backup passers in Superflex, and injured players whose
              landing spot still makes sense. Do not take a 28-year-old free agent running
              back &ldquo;because he might get signed.&rdquo; That is redraft thinking.
            </p>
            <div class="highlight-box">
              Choose a window, honor the format, and spend the middle rounds on roles, not
              names. The startup is a roster, not a celebrity list.
            </div>
        """,
    },
    "waiver-wire-and-faab": {
        "title": "Waiver Wire and FAAB Strategy for Fantasy Football",
        "summary": "How to spend a FAAB budget, when to use waiver priority, and which "
                   "pickups actually change a roster.",
        "published": GUIDE_UPDATED,
        "updated": GUIDE_UPDATED,
        "body": """
            <p>
              The waiver wire is the only market in fantasy where you compete against the
              whole league every week without needing someone to say yes. In redraft it is
              how championships are patched together. In dynasty it is how you find the
              cheap young player who becomes next year's trade chip. The mechanics differ
              (priority list vs FAAB budget), but the evaluation does not: you are bidding
              on opportunity, not on last Thursday's box score.
            </p>
            <div class="static-section-title">FAAB vs waiver priority</div>
            <p>
              FAAB (Free Agent Acquisition Budget) is a season-long pile of dollars you
              bid in secret. Hitting 100% of your budget in week three on a backup who
              lost the job by week six is a classic mistake. Hitting 0% until December
              and watching every useful add go to more aggressive managers is the opposite
              mistake. A simple split: save a real bid (often 15-40% depending on league
              size) for a genuine role change, and use small bids to win uncontested
              adds.
            </p>
            <p>
              Rolling priority rewards patience and punishes panic. If your league uses
              it, do not burn the top claim on a player you could have had for a
              mid-priority add. Save the claim for a starter-level role that will not
              last more than a week on waivers.
            </p>
            <div class="static-section-title">What is actually worth a claim</div>
            <ul style="margin-left:20px; line-height:1.8;">
              <li>A running back who just inherited a backfield because of injury, with
                  the snaps to match, not just a highlight carry.</li>
              <li>A receiver or tight end whose target share jumped after a depth-chart
                  change, which you can verify in
                  <a href="/guides/reading-advanced-metrics">advanced metrics</a>.</li>
              <li>In Superflex, a quarterback who is suddenly starting. Format scarcity
                  makes streaming passers more valuable than in 1QB.</li>
              <li>In dynasty, a young player who just earned a real NFL role even if the
                  short-term points are modest. That is future
                  <a href="/guides/dynasty-trade-value">trade value</a>.</li>
            </ul>
            <p>
              What is rarely worth a big bid: a one-week dart throw with no path to snaps
              once the starter returns, a player you will not start or stash, and a name
              from a national recap that every other manager also saw.
            </p>
            <div class="static-section-title">Dynasty-specific waiver habits</div>
            <p>
              Taxi squads change the math. A rookie who is not startable this year can
              still be a correct add if he has draft capital and a path. Do not clog
              taxi with 28-year-old lottery tickets. Prefer the same traits you would
              want in a
              <a href="/guides/rookie-draft-strategy">rookie draft</a>: age, capital,
              and opportunity.
            </p>
            <p>
              BR Fantasy's waiver tools rank pickups against your roster's needs, scoring,
              and remaining schedule. Use that as a shortlist, then check snaps and
              targets before you spend. The wire rewards managers who move on Tuesday
              with a plan, not managers who bid on every trending name.
            </p>
            <div class="highlight-box">
              Bid on roles that will still exist next month. Spend real FAAB rarely, and
              never spend it on a headline without usage behind it.
            </div>
        """,
    },
    "adp-vs-trade-value": {
        "title": "ADP vs Trade Value: Why Draft Price Is Not Trade Price",
        "summary": "Startup ADP and dynasty trade values answer different questions. "
                   "Here's how to use both without mixing them up.",
        "published": GUIDE_UPDATED,
        "updated": GUIDE_UPDATED,
        "body": """
            <p>
              Average Draft Position (ADP) is a record of where players have been taken in
              recent drafts. Dynasty trade value is an estimate of what those players are
              worth in trades right now. They are related, and they often disagree, and
              that disagreement is useful if you know which question you are answering.
            </p>
            <div class="static-section-title">What ADP is good for</div>
            <p>
              ADP is a clock. In a startup, it tells you the latest you can reasonably wait
              on a player before someone else takes him. In a rookie draft, it tells you
              which names are likely gone at the turn. It is also a measure of consensus
              hype, which is helpful when you want to fade a crowded name or when you want
              to know that your league will pay up for a certain quarterback.
            </p>
            <p>
              ADP is a weak measure of true talent. Draft rooms are public, social, and
              slow to update after injuries and depth-chart news. A player can climb ADP
              for three weeks on a narrative while his snap share does not move. That is
              why BR Fantasy treats market data as one input into
              <a href="/guides/dynasty-trade-value">trade value</a>, not the whole model.
            </p>
            <div class="static-section-title">What trade value is good for</div>
            <p>
              Trade value is for deals, keep/cut decisions, and comparing two players who
              will never share a draft room again. It can move daily with production and
              market trades. If you are asking &ldquo;should I accept this package?&rdquo;
              you want the
              <a href="/dynasty-trade-value-chart">value chart</a> and the
              <a href="/trade">trade calculator</a>, not last month's startup ADP.
            </p>
            <p>
              Trade value is a weak start/sit tool. A high dynasty number can belong to a
              young player you should bench this week. Do not start the expensive name
              over the boring volume earner because the chart said so.
            </p>
            <div class="static-section-title">When they diverge, look for a reason</div>
            <ul style="margin-left:20px; line-height:1.8;">
              <li><strong>ADP higher than trade value:</strong> the player is a draft-room
                  celebrity. Fine to let him go in a startup if your board prefers a
                  less-famous role player. Risky to buy him in-season at ADP prices.</li>
              <li><strong>Trade value higher than ADP:</strong> the market of managers who
                  already own him will not sell cheap, even if drafters have not caught
                  up. That often happens after a mid-season breakout. See
                  <a href="/top-movers">top movers</a>.</li>
              <li><strong>Both falling:</strong> believe the role change until metrics
                  say otherwise. Do not &ldquo;buy the name&rdquo; just because ADP used
                  to be high.</li>
            </ul>
            <p>
              Rookie ADP is especially noisy before the NFL draft and again after training
              camp. Use
              <a href="/prospects">prospect profiles</a> and landing-spot context, not
              just a big board screenshot from May.
            </p>
            <div class="static-section-title">How BR Fantasy uses both</div>
            <p>
              The value model treats consensus market data, including draft prices, as one
              input among several. That is why a player can sit above or below his ADP on
              the <a href="/dynasty-trade-value-chart">chart</a> after a role change the
              draft community has not fully priced. If you only ever draft by ADP, you
              will own the consensus team. If you only ever trade by ADP, you will
              mis-price in-season movers.
            </p>
            <p>
              When you are unsure, start with trade value for deals and ADP for
              &ldquo;will he be there in two rounds?&rdquo; That split keeps
              <a href="/guides/startup-draft-guide">startup drafts</a> and mid-season
              trades from using the same number for two different jobs.
            </p>
            <div class="highlight-box">
              Use ADP to time a draft pick. Use trade value to judge a deal. Mixing the
              two is how you overpay in March and under-sell in October.
            </div>
        """,
    },
    "positional-aging-curves": {
        "title": "Positional Aging Curves in Dynasty",
        "summary": "Why running backs fall off earlier than receivers and quarterbacks, "
                   "and how to stop treating every 27-year-old the same.",
        "published": GUIDE_UPDATED,
        "updated": GUIDE_UPDATED,
        "body": """
            <p>
              Dynasty value is a bet on remaining seasons, not just remaining weeks. That
              is why two players with similar projections this year can sit in completely
              different tiers on a
              <a href="/dynasty-trade-value-chart">trade value chart</a>. Aging is not a
              cliff you notice on a birthday. It is a slow change in injury risk, role,
              and how NFL teams choose to use a player. The curve is different at every
              position.
            </p>
            <div class="static-section-title">Running backs: the steepest curve</div>
            <p>
              Running backs take the most wear and have the shortest peak. Many still
              produce at 26 and 27; far fewer are weekly RB1s at 29. That does not mean
              you should never roster an aging back. It means their correct price is a
              rental: valuable to a
              <a href="/guides/contending-in-dynasty">contender</a>, dangerous as a
              cornerstone in a
              <a href="/guides/dynasty-rebuild-strategy">rebuild</a>.
            </p>
            <p>
              When an older back is having a career year, that is often the
              <a href="/guides/buy-low-sell-high">sell-high</a> window, not the time to
              extend your window around him. Check whether the production is coming from
              a huge snap share you cannot count on next year.
            </p>
            <div class="static-section-title">Receivers: a longer prime</div>
            <p>
              Wide receivers as a group hold production deeper into their late 20s,
              especially high-volume players who do not rely on elite speed alone. A
              27-year-old WR1 with sticky target share is still a building block. A
              27-year-old gadget receiver who lives on big plays is not the same aging
              profile just because the birthdays match.
            </p>
            <p>
              Young receivers can take a year to earn targets. Paying up for a 22-year-old
              with draft capital and a clear path is usually better than paying up for a
              24-year-old who has already failed to earn a role. Metrics help separate
              those cases; see
              <a href="/guides/reading-advanced-metrics">reading advanced metrics</a>.
            </p>
            <div class="static-section-title">Quarterbacks: format decides the price</div>
            <p>
              Passers age more slowly than skill players, which is why veterans remain
              useful even as their dynasty number cools. In Superflex, a 32-year-old
              starter can still be a high-value asset because the replacement is so
              expensive. In 1QB, that same player is often a cheap stabilizer. Age still
              matters for the true elites you would build around for five years; it
              matters less for the QB12 you are starting this season. Details in
              <a href="/guides/superflex-vs-1qb">Superflex vs 1QB</a>.
            </p>
            <div class="static-section-title">Tight ends: late bloomers, then a cliff</div>
            <p>
              Tight ends often break out later than receivers. Patience in year two and
              three is rational, especially in
              <a href="/guides/te-premium-leagues">TE premium</a>. Once they arrive, they
              can hold value into their late 20s. The players to be careful with are
              28-plus tight ends whose role is blocking-first with occasional scoring
              spikes. Those spikes are redraft candy and dynasty traps.
            </p>
            <p>
              BR Fantasy's values bake positional aging into the daily model so you do
              not have to apply a homemade discount in every trade. You still have to
              decide whether <em>your</em> window matches the player's remaining prime.
            </p>
            <div class="highlight-box">
              Age is position-specific. Price running backs as rentals, receivers as
              cores, and Superflex quarterbacks as scarce seasons, not birthdays.
            </div>
        """,
    },
    "using-the-trade-calculator": {
        "title": "How to Use the BR Fantasy Trade Calculator",
        "summary": "A practical walkthrough of grading a deal, toggling league settings, "
                   "and turning a lopsided offer into a counter that might actually land.",
        "published": GUIDE_UPDATED,
        "updated": GUIDE_UPDATED,
        "body": """
            <p>
              The <a href="/trade">trade calculator</a> is the public tool most managers
              hit first: put players (and picks) on two sides, see how the values compare,
              and decide whether to accept, reject, or counter. It is not trying to
              replace your judgment. It is trying to stop you from negotiating from a
              screenshot of last year's rankings.
            </p>
            <div class="static-section-title">Match the calculator to your league</div>
            <p>
              Before you add names, set the format. Superflex vs 1QB changes quarterback
              prices more than any other toggle. Team count and scoring (including TE
              premium) also move the result. If those settings are wrong, the grade is
              wrong, and you will either insult a league-mate with a lowball or accept a
              deal that only looks fair on a default chart. Background on why the
              numbers differ is in
              <a href="/guides/dynasty-trade-value">how dynasty trade value works</a>
              and <a href="/guides/superflex-vs-1qb">Superflex vs 1QB</a>.
            </p>
            <p>
              If you have already connected a league, the calculator can use your roster
              context. That is the difference between a generic &ldquo;side A wins by
              8%&rdquo; and a deal that actually fills a hole you have. Connecting is
              optional for a quick public look; it is worth it if you trade often.
            </p>
            <div class="static-section-title">Read the grade, then ignore it slightly</div>
            <p>
              A balanced grade means the market thinks the packages are close, not that
              the trade is right for you. A lopsided grade means you should have a reason
              to continue: consolidation, a win-now running back, or a rebuild unpack.
              Those reasons are spelled out in
              <a href="/guides/evaluating-a-trade">how to evaluate a dynasty trade</a>.
              If you cannot name one, the grade is the answer.
            </p>
            <p>
              Counters exist because first offers are rarely final. If the calculator
              shows you giving up a large value gap, add a pick or swap a player rather
              than sending a message that the other side is &ldquo;crazy.&rdquo; Deals
              close when both managers can defend the result in the group chat.
            </p>
            <div class="static-section-title">Use the rest of the site around it</div>
            <ul style="margin-left:20px; line-height:1.8;">
              <li>Check whether a name is rising or falling on
                  <a href="/top-movers">top movers</a> before you buy a dip that is
                  actually a lost role.</li>
              <li>Confirm usage with
                  <a href="/guides/reading-advanced-metrics">advanced metrics</a>
                  instead of the last box score.</li>
              <li>Look up a single player's page if you want the long-form value
                  context, then come back to the two-sided grade.</li>
            </ul>
            <p>
              The calculator cannot see that your league never trades first-round picks
              in-season, or that one manager will not sell a certain player at any
              price. Those are table-stakes of your league. Use the tool for the market
              half of the decision, then be a human for the rest.
            </p>
            <div class="static-section-title">Picks, multi-player deals, and counters</div>
            <p>
              Include rookie picks on the side that is actually sending them. A
              &ldquo;player for player&rdquo; grade that forgets the extra second-round
              pick is how people think they won a trade that the rest of the league
              immediately calls a fleece. If the first grade is lopsided, add the
              smallest piece that closes the gap rather than swapping the entire
              package.
            </p>
            <p>
              For a longer framework that sits on top of the grade, use
              <a href="/guides/evaluating-a-trade">how to evaluate a dynasty trade</a>.
              The calculator is the scale. That guide is the recipe.
            </p>
            <div class="highlight-box">
              Set the format correctly, treat the grade as a first check, and only
              override it when the deal fits your window.
            </div>
        """,
    },
}

GUIDE_ORDER = [
    "dynasty-trade-value",
    "superflex-vs-1qb",
    "reading-advanced-metrics",
    "rookie-draft-strategy",
    "buy-low-sell-high",
    "evaluating-a-trade",
    "dynasty-rebuild-strategy",
    "contending-in-dynasty",
    "te-premium-leagues",
    "startup-draft-guide",
    "waiver-wire-and-faab",
    "adp-vs-trade-value",
    "positional-aging-curves",
    "using-the-trade-calculator",
]
