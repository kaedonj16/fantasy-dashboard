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
GUIDE_UPDATED = "2026-10-02"

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
              single number meant to capture what that player is worth in the open market
              of league-to-league trades. It is not a redraft ranking. A redraft ranking
              answers &ldquo;who scores the most points this season?&rdquo; A dynasty value
              answers &ldquo;what would the rest of the league actually give up to acquire
              this player, accounting for age, contract, expected production, and
              long-term outlook?&rdquo;
            </p>
            <p>
              That distinction is why a 23-year-old breakout receiver can out-value a
              30-year-old running back who scores more points <em>right now</em>.
              Dynasty rosters are held for years, so the market prices in the runway a
              player has left, not just this week&rsquo;s box score.
            </p>
            <h2 class="static-section-title">The market, not the ranking</h2>
            <p>
              Redraft rankings are ordinal: they list players in order. Dynasty value is
              cardinal: a number you can add, subtract, and compare across positions. A
              ranking can call a 25-year-old receiver WR14 and a 29-year-old WR15 while
              the values read 40 and 24. The ranking shrugs at the gap; the values say
              one costs nearly twice as much.
            </p>
            <p>
              Cardinal numbers are what make trades work: a 40 plus a 12 on one side
              against a 30 plus a 20 on the other, a rookie pick priced the same way as
              a veteran. None of that is possible with &ldquo;WR14 versus WR15.&rdquo;
            </p>
            <p>
              The common beginner mistake is treating a value as a verdict on who is
              &ldquo;better.&rdquo; It is a price tag written by the market. A 24-year-old
              WR2 coming off back-to-back top-20 finishes might carry a 45 on three or
              four projected good years, while a 28-year-old RB with one year left of
              elite production carries a 38 despite outscoring him weekly. The market is
              charging less for a shorter runway, not calling the running back worse.
            </p>
            <h2 class="static-section-title">What goes into a value</h2>
            <p>
              A good dynasty value blends several inputs rather than relying on any
              single source:
            </p>
            <ul style="margin-left:20px; line-height:1.8;">
              <li><strong>Consensus market data</strong>: where the crowd of dynasty
                  managers is actually pricing a player. Startup ADP, recent trade
                  results, and public rankings all describe a market, but none of them
                  is the market alone. See
                  <a href="/guides/adp-vs-trade-value">ADP vs trade value</a>.</li>
              <li><strong>Recent on-field production</strong>: usage, efficiency, and
                  role, which move a player&rsquo;s stock week to week. A three-game
                  spike in yards is less informative than a three-game spike in snaps
                  and targets. Yards can be luck. Snaps are a job.</li>
              <li><strong>Age and position curve</strong>: running backs decline early,
                  while wide receivers and quarterbacks hold value far longer. A
                  27-year-old WR and a 27-year-old RB are not in the same phase of
                  their careers. Details in
                  <a href="/guides/positional-aging-curves">positional aging curves</a>.</li>
              <li><strong>Situation</strong>: target share, depth-chart competition, and
                  team context that shape future opportunity. A talented player in a
                  crowded room is priced differently from the same talent with a clean
                  path to snaps.</li>
            </ul>
            <h2 class="static-section-title">Calibrated daily, guarded against noise</h2>
            <p>
              BR Fantasy recalculates these inputs daily and calibrates against
              observed dynasty prices, with guardrails so one noisy week cannot send a
              veteran from a third-round startup pick to a first. A single 30-point game
              moves the needle; it does not rebuild it.
            </p>
            <p>
              The guardrails matter most in two spots. Veterans should not lose a fifth
              of their value on one bad game, because the market does not trade them
              that way. Breakouts should move, but the full repricing waits for the
              role behind the box score: rising snap share and target share over
              several weeks. The <a href="/breakouts">breakout board</a> tracks that gap
              between production and opportunity.
            </p>
            <p>
              Browse calibrated values on the
              <a href="/rankings/dynasty">dynasty rankings</a> page or the full
              <a href="/dynasty-trade-value-chart">dynasty trade value chart</a>.
            </p>
            <h2 class="static-section-title">How to read a number on the chart</h2>
            <p>
              A value is a <em>relative</em> price, not a prediction of next
              week&rsquo;s points. Compare players to each other, not to a fantasy of
              what the number &ldquo;should&rdquo; be. A 42 and a 28 is a meaningful
              gap; a 42.1 and a 41.8 is noise. Round ruthlessly and save your attention
              for gaps that are actually tradeable.
            </p>
            <p>
              When two players sit in the same band, the decision is about fit: age,
              position need, and whether you are
              <a href="/guides/contending-in-dynasty">contending</a> or
              <a href="/guides/dynasty-rebuild-strategy">rebuilding</a>. A 36-valued
              receiver and a 36-valued running back are the same price and completely
              different purchases: the contender buys the running back&rsquo;s next ten
              games, the rebuilder buys the receiver&rsquo;s next four years.
            </p>
            <p>
              Watch movement, not just the snapshot. The
              <a href="/top-movers">top movers</a> page is where a value becomes a
              trading window: a player falling on usage is a different kind of dip than
              one who scored 4 points on 90% of snaps. Check
              <a href="/guides/reading-advanced-metrics">advanced metrics</a> before
              you decide the market is wrong.
            </p>
            <h2 class="static-section-title">Format and scoring move the number</h2>
            <p>
              Most dynasty leagues are either single-quarterback (1QB) or Superflex,
              and a player&rsquo;s value can change dramatically between formats.
              Quarterbacks are far more valuable in Superflex because you can start two
              of them, so always use Superflex values there: 1QB numbers badly
              under-rate every passer. We cover this in depth in
              <a href="/guides/superflex-vs-1qb">Superflex vs 1QB</a>.
            </p>
            <p>
              Scoring moves the number too. Tight-end premium, full PPR, and long-score
              bonuses change which archetypes the market pays up for. A value is only
              useful if it was built for a league like yours: if your league pays extra
              for tight ends, start with
              <a href="/guides/te-premium-leagues">TE premium strategy</a> before
              treating a 1-PPR chart as gospel.
            </p>
            <h2 class="static-section-title">A worked example: pricing an offer</h2>
            <p>
              Say you are a contender and a league-mate offers his 22-year-old WR3,
              whose snap share has climbed five straight weeks, plus a mid first-round
              rookie pick, for your 28-year-old RB with one year left of elite
              production. Check both sides on the
              <a href="/dynasty-trade-value-chart">value chart</a> with your
              format&rsquo;s settings: if the running back is a 38, the young receiver
              a 30, and a mid first a 25, the raw math favors the package.
            </p>
            <p>
              Second, check direction on <a href="/top-movers">top movers</a> and the
              role on the <a href="/metrics">advanced metrics page</a>: is the receiver
              rising on usage or on one touchdown-heavy game? Third, match the deal to
              your timeline. The contender takes the package only if the receiver can
              start this year; the rebuilder takes it even if the raw math were
              slightly short, because the running back&rsquo;s 38 expires with the
              season.
            </p>
            <h2 class="static-section-title">What a value is not</h2>
            <p>
              It is not a projection, a start/sit grade, or a guarantee that a trade
              will &ldquo;win.&rdquo; It does not know you already have three receivers
              and zero running backs, or that your league mates refuse to trade
              first-round picks in-season. That is why the
              <a href="/trade">trade calculator</a> exists: it applies the same values
              to both sides of a deal and leaves room for you to judge fit. For a
              step-by-step process, see
              <a href="/guides/evaluating-a-trade">how to evaluate a dynasty trade</a>.
            </p>
            <div class="highlight-box">
              Bottom line: a dynasty value is a market estimate, not a law. Use it as
              the starting point for a negotiation, then adjust for your roster&rsquo;s
              timeline and needs.
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
              The single biggest factor in a player&rsquo;s dynasty value, bigger than age,
              bigger than last week&rsquo;s stat line, is often just your league format.
              In a <strong>single-quarterback (1QB)</strong> league you start one QB. In a
              <strong>Superflex</strong> league you can start a second quarterback in a flex
              spot, which makes the position enormously more valuable.
            </p>
            <h2 class="static-section-title">Why quarterbacks explode in Superflex</h2>
            <p>
              There are only 32 starting NFL quarterbacks, and in a 12-team Superflex
              league up to 24 of them can be in starting lineups every week. That
              scarcity means even mid-tier starters carry real weight, and the elite
              young passers become the most valuable assets in the entire player pool,
              frequently worth more than any running back or receiver.
            </p>
            <p>
              The reason is replacement level. When 24 passers are starting, the waiver
              wire is roughly the 25th-best quarterback, and he scores meaningfully
              fewer points than the 18th. Every startable passer you roster is points
              your opponents cannot have. That is why Superflex startup drafts see
              quarterbacks fly off the board in a way that shocks 1QB-only managers.
            </p>
            <h2 class="static-section-title">The 1QB world: passers are replaceable</h2>
            <p>
              In 1QB, the opposite is true: you only need one quarterback, streamable
              options are everywhere, and so the position is heavily discounted. Top-tier
              wide receivers and running backs sit at the top of 1QB value charts instead.
            </p>
            <p>
              This is not disrespect for the position. It is math. If only 12 quarterbacks
              start and the 13th through 18th sit on benches and waivers scoring within
              a few points of the 10th, then paying a premium for QB5 over QB14 buys you
              very little. The same draft capital spent on a young WR1 buys a positional
              advantage that actually shows up in the standings.
            </p>
            <h2 class="static-section-title">The scarcity math, in plain terms</h2>
            <p>
              If 12 teams each start one quarterback, the 13th-best passer is a backup.
              If 12 teams each start two, the 20th-best passer is still a weekly starter.
              That second cohort, the passers ranked roughly 18th to 24th in a given
              year, is where Superflex leagues are won and 1QB leagues barely notice.
            </p>
            <p>
              Think of it as a cliff. In 1QB, the cliff sits behind QB12 and almost
              nobody cares, because everyone already has a starter. In Superflex, the
              cliff sits behind QB24 and everyone cares, because the team starting QB25
              in the flex is spotting the field several points a week. That weekly
              deficit is why the market prices the 20th-best passer like a low-end WR1
              in Superflex and like a bench stash in 1QB.
            </p>
            <h2 class="static-section-title">The practical trap: the wrong format&rsquo;s numbers</h2>
            <p>
              The most common dynasty trade mistake is using the wrong format&rsquo;s
              values. Picture a Superflex league where you are offered a young WR2 for
              your QB2, a passer in that 18-to-24 band. On a 1QB chart, the receiver
              might be a 30 and the quarterback an 18. You would think you are winning
              the deal.
            </p>
            <p>
              On a Superflex chart, that same quarterback might be a 34. You would be
              trading a weekly starter at the scarcest position for a flex receiver,
              and your lineup would have a hole at the exact spot that is hardest to
              fill. Always confirm which format a value reflects before you commit. On
              the <a href="/rankings/dynasty">dynasty rankings</a> you can view values
              for the format your league uses, and the
              <a href="/trade">trade calculator</a> lets you toggle Superflex so both
              sides of a deal are priced correctly.
            </p>
            <h2 class="static-section-title">How the rest of the board moves</h2>
            <p>
              Superflex does not only inflate quarterbacks. It compresses everyone
              else. Elite running backs and receivers still matter, but they occupy a
              smaller share of total league value because so much of the pie is locked
              in the passer market. In practical terms:
            </p>
            <ul style="margin-left:20px; line-height:1.8;">
              <li>A top-five Superflex quarterback often costs what a top-three 1QB
                  skill player costs. Do not &ldquo;feel&rdquo; that as a rip-off; it
                  is the format working.</li>
              <li>Mid-round startup running backs are relatively cheaper in Superflex,
                  because managers spent early capital on passers. That is a feature
                  if you are
                  <a href="/guides/startup-draft-guide">building a startup</a> around
                  depth.</li>
              <li>Rookie firsts still matter, but a late first that projects as a
                  quarterback with a path to starting is Superflex gold and 1QB noise.
                  See <a href="/guides/rookie-draft-strategy">rookie draft strategy</a>.</li>
              <li>Veteran passers hold value longer in Superflex: a 32-year-old starter
                  is a real asset because his replacement costs a fortune. In 1QB he is
                  a cheap stabilizer. More on the age side in
                  <a href="/guides/positional-aging-curves">positional aging curves</a>.</li>
            </ul>
            <h2 class="static-section-title">Roster construction and in-season trading</h2>
            <p>
              In Superflex, quarterback depth is not a luxury, it is insurance. One
              injury can turn a contender into a team starting a backup in the flex.
              Holding three startable passers is a common, rational strategy; holding
              five is usually dead capital unless you plan to sell into a QB-starved
              league. In 1QB, a second quarterback is a handcuff, not a cornerstone.
            </p>
            <p>
              Quarterbacks are also trade currency. In a Superflex league where two or
              three teams are starting shaky QB2s, your third startable passer is worth
              more in trade than on your bench. Shop him to the desperate, not to the
              comfortable. And when you are offered a &ldquo;fair&rdquo; deal that sends
              your QB2 for a young receiver, ask what your lineup looks like in week 14
              if your QB1 misses time. The
              <a href="/guides/evaluating-a-trade">trade evaluation process</a> exists
              for exactly this kind of format-aware check, not just adding the numbers.
            </p>
            <h2 class="static-section-title">Quick checklist before any quarterback trade</h2>
            <ul style="margin-left:20px; line-height:1.8;">
              <li><strong>Confirm the format toggle</strong> on every number you cite:
                  <a href="/trade">trade calculator</a>,
                  <a href="/dynasty-trade-value-chart">value chart</a>, and
                  <a href="/rankings/dynasty">rankings</a> all need to match your
                  league.</li>
              <li><strong>Count your startable passers after the deal</strong>, not just
                  the value gained. In Superflex, dropping from three to two is a risk
                  you should get paid for.</li>
              <li><strong>Price the second cohort, not the stars.</strong> The 18-to-24
                  passers are where most Superflex trades are actually decided.</li>
              <li><strong>Match the move to your window.</strong> A
                  <a href="/guides/contending-in-dynasty">contender</a> rents veteran
                  passers; a
                  <a href="/guides/dynasty-rebuild-strategy">rebuilder</a> collects
                  young ones with paths. For how the numbers themselves get built, see
                  <a href="/guides/dynasty-trade-value">how dynasty trade value works</a>.</li>
            </ul>
            <div class="highlight-box">
              Rule of thumb: in Superflex, treat startable quarterbacks as premium
              assets. In 1QB, let the other manager overpay for them.
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
              Box-score stats tell you what already happened. <strong>Advanced
              metrics</strong> tell you whether it is likely to keep happening. They
              separate players who are producing because of genuine, repeatable
              opportunity from those riding unsustainable efficiency or touchdown luck.
            </p>
            <p>
              Picture two receivers after week 4. One just posted a 40-point game but
              owns a 12% target share on the season. The other has never cracked 15
              points but owns a 30% target share. The box score says the first player
              is better. The metrics say the second one is the player to own. Here is
              how to read the numbers that matter.
            </p>
            <h2 class="static-section-title">The one mental model: opportunity is sticky, efficiency is not</h2>
            <p>
              Almost every advanced metric falls into one of two buckets.
              <strong>Opportunity metrics</strong> describe what a player is asked to
              do: snaps, routes, targets, carries, red-zone looks.
              <strong>Efficiency metrics</strong> describe how well he did it: yards per
              touch, yards per route run, catch rate, yards after catch.
            </p>
            <p>
              Opportunity is mostly a coaching decision, and coaching decisions repeat.
              A player running 90% of his team&rsquo;s routes through four weeks is very
              likely to run a similar share in week 5. Efficiency is mostly a product
              of defense, game script, and luck, and it bounces constantly. A back can
              average 6.5 yards per carry for a month and 3.9 the next with no change in
              his role.
            </p>
            <p>
              The hierarchy is simple: weight opportunity heavily, treat efficiency as
              context, and distrust any hot streak that is not backed by a role.
              Everything below is that principle applied to the metrics you will
              actually use.
            </p>
            <h2 class="static-section-title">Opportunity metrics: the five to memorize</h2>
            <p>
              If you only track five numbers, make them these:
            </p>
            <ul style="margin-left:20px; line-height:1.8;">
              <li><strong>Target share.</strong> The percentage of his team&rsquo;s
                  targets a receiver or tight end earns. A share climbing from 15% to
                  25% over a month is a role being handed to someone in real time, and
                  one of the strongest leading indicators in fantasy.</li>
              <li><strong>Snap share and route participation.</strong> How often the
                  player is on the field, and for receivers, how often he is actually
                  running routes. Low snap share caps a ceiling no matter how efficient
                  a player looks. A part-time player cannot be a full-time producer.</li>
              <li><strong>Air yards.</strong> The total downfield distance of a
                  player&rsquo;s targets. High air yards with low catch totals means a
                  high-value role before the catches show up. It is one of the earliest
                  signals that a breakout is coming.</li>
              <li><strong>Red-zone usage.</strong> Touches and targets inside the 20.
                  Red-zone volume drives touchdowns, the most volatile and most valuable
                  source of fantasy points. A player earning red-zone looks every week
                  is one matchup away from a spike.</li>
              <li><strong>Backfield share.</strong> A running back&rsquo;s share of his
                  team&rsquo;s carries plus targets. Count touches, not just carries: a
                  back with 12 carries and 5 targets is in a better role than a back
                  with 15 carries and none.</li>
            </ul>
            <h2 class="static-section-title">Efficiency metrics: useful context, never gospel</h2>
            <p>
              Yards per route run, yards after catch, yards per touch, and catch rate
              describe how well a player converts opportunity into production. They are
              useful for exactly one job: telling you whether a player&rsquo;s scoring
              pace is supported by how he plays, or whether it is a house built on sand.
            </p>
            <p>
              The failure mode is reading efficiency as talent. A running back who is
              95th percentile in yards per carry but 20th percentile in snap share is
              not a league-winner in disguise. He is a change-of-pace back having a hot
              month on limited touches, and the moment his workload grows, the
              efficiency almost always falls back toward average. Flip it: a receiver
              at the 40th percentile in yards per route run but the 90th percentile in
              target share is a volume earner whose floor is safer than the
              efficiency-first scouts admit. Buy the role, not the rate.
            </p>
            <p>
              Catch rate looks like a skill stat but is mostly a role stat: slot
              receivers will always catch more than deep threats, and a falling catch
              rate often just means deeper targets, which is good for fantasy. Compare
              within a role and within a player&rsquo;s own history, never across roles.
            </p>
            <h2 class="static-section-title">The two patterns that make decisions</h2>
            <p>
              You do not need twenty metrics. You need to recognize two shapes in the
              data.
            </p>
            <p>
              <strong>Rising opportunity, lagging output: buy.</strong> Snaps climbing,
              target share climbing, red-zone role growing, but the fantasy points have
              not caught up yet. This is the cheapest a good player ever gets, because
              your league-mates are still pricing the box score. It is exactly the gap
              the <a href="/breakouts">breakout engine</a> is built to surface, and you
              can dig into the underlying numbers for any player on the
              <a href="/players">player database</a>.
            </p>
            <p>
              <strong>Output running ahead of opportunity: sell.</strong> The points are
              there but the snaps, targets, and red-zone looks are not. This is the
              classic <a href="/guides/buy-low-sell-high">sell-high</a> candidate.
              Touchdowns cluster, markets overfit to the last four games, and the
              metrics keep you honest when the box score is screaming that a part-time
              player is a star.
            </p>
            <h2 class="static-section-title">Sample size: when a trend is tradeable</h2>
            <p>
              One week of 22% target share is a headline. Six weeks of 22% target share
              is a role. Early-season usage is noisy because coaching staffs are still
              sorting personnel, so by week six snap and target shares have usually
              settled enough to trade on with confidence. At week 4, treat the numbers
              as directional: act on two or three weeks of a trend if the depth-chart
              story matches, but do not pay full price for a one-week spike.
            </p>
            <p>
              Two things reset the clock. A new starter&rsquo;s first two games
              outweigh a veteran&rsquo;s quiet week in a blowout, because the role
              itself is new. And game script: blowouts inflate passing volume for the
              trailing team, so confirm a target spike came in competitive game script
              before treating it as a role change.
            </p>
            <p>
              Age still sits underneath every metric. A 24-year-old with rising snaps
              is a different bet from a 29-year-old with the same chart, because
              <a href="/guides/positional-aging-curves">positional aging curves</a> say
              the younger player can still grow into the role. Metrics describe the
              present. Dynasty value has to price the future.
            </p>
            <h2 class="static-section-title">Five mistakes to stop making</h2>
            <ul style="margin-left:20px; line-height:1.8;">
              <li><strong>Trading on one week.</strong> A single-game spike with no role
                  change behind it is noise. Wait for the second data point, or pay a
                  noise price, not a role price.</li>
              <li><strong>Buying efficiency without volume.</strong> Elite per-touch
                  numbers on five touches a game do not survive a larger sample. If the
                  role does not grow, the rate will fall.</li>
              <li><strong>Ignoring the depth chart.</strong> Metrics describe what a
                  player did in his old role. If he was just promoted, or just lost his
                  job to a returning starter, the old averages describe a situation
                  that no longer exists.</li>
              <li><strong>Comparing across roles.</strong> Slot receivers, deep
                  threats, pass-catching backs, and goal-line backs live in different
                  statistical neighborhoods. Compare a player to his own history and
                  his direct role peers.</li>
              <li><strong>Forgetting the offense.</strong> A 25% target share in a
                  pass-heavy attack beats the same share in a run-first offense.
                  Reattach player metrics to team context before you pay up.</li>
            </ul>
            <div class="highlight-box">
              Prioritize volume and role over efficiency. Opportunity is sticky,
              efficiency regresses, and the managers who trade on the first one beat
              the managers who chase the second.
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
              The rookie draft is where dynasty championships are quietly built. Cheap,
              ascending young talent is the best value in the format: a rookie who hits
              gives you a starter on a cost-controlled roster spot for years. But rookie
              picks are also where managers most often overpay for hype, turning proven
              production into lottery tickets at exactly the wrong exchange rate.
            </p>
            <p>
              A good rookie draft is not about picking the best players. It is about
              knowing what each pick is worth, what actually predicts success, and
              when the right move is to trade the pick instead of making it.
            </p>
            <h2 class="static-section-title">Value the picks, then the players</h2>
            <p>
              Before you fall in love with a prospect, understand what the pick itself
              is worth. Early first-round rookie picks carry significant trade value
              because of their upside, but that value drops quickly into the second
              and third rounds. A late first is roughly a mid-tier veteran. Knowing
              the market price of a pick keeps you from trading a proven player for a
              lottery ticket.
            </p>
            <p>
              A useful habit: price the pick on your
              <a href="/dynasty-trade-value-chart">trade value chart</a> the way you
              would price a player. If a late first is worth roughly a mid-tier
              veteran, do not spend a league-winning starter to &ldquo;get
              younger&rdquo; unless the prospect&rsquo;s median outcome actually beats
              that veteran on your timeline.
            </p>
            <p>
              Future firsts deserve their own discipline. Next year&rsquo;s first is an
              option on a player who does not exist yet, and its value depends
              entirely on where it lands. Price it as a mid first unless you have real
              reason to think it will be early or late. Selling a future first to patch
              a one-week hole is how contenders quietly become average.
            </p>
            <h2 class="static-section-title">What actually predicts rookie success</h2>
            <p>
              Prospect evaluation is noisy, but four signals carry most of the weight:
            </p>
            <ul style="margin-left:20px; line-height:1.8;">
              <li><strong>Draft capital</strong>: where the NFL drafted a player is one
                  of the best predictors of opportunity. Teams invest snaps and targets
                  in the players they spent premium picks on. A second-round receiver
                  with a path to snaps beats a fifth-round receiver with a highlight
                  reel.</li>
              <li><strong>Landing spot</strong>: the same prospect can be a
                  league-winner or a redraft afterthought depending on depth-chart
                  competition and offensive quality. A talented receiver buried behind
                  two entrenched veterans is a taxi-squad stash. Read the depth chart,
                  not the scouting report.</li>
              <li><strong>College production at a young age</strong>: players who
                  dominated early in their college careers (a strong &ldquo;breakout
                  age&rdquo;) hit at higher rates than older players who needed three
                  years to produce against the same competition.</li>
              <li><strong>Athletic profile</strong>: testing scores like RAS provide a
                  floor check, especially at receiver and running back. They do not
                  predict stardom, but players below certain athletic thresholds
                  almost never become every-down NFL contributors. Use athleticism to
                  eliminate, not to fall in love.</li>
            </ul>
            <p>
              You can study all of it on the
              <a href="/prospects">rookie prospects</a> page: college metrics, draft
              capital, athletic scores, and live ADP movement in one place.
            </p>
            <h2 class="static-section-title">Position priorities, by window</h2>
            <p>
              In most formats, prioritize wide receivers early. They have the longest
              dynasty shelf life and the highest hit rate near the top of rookie
              drafts. A receiver who hits in year one or two gives you five-plus
              seasons of production, which is exactly the asset a rookie pick is
              supposed to buy.
            </p>
            <p>
              Running backs offer immediate production but age out fast, so target
              them when you are contending. Rebuilders should stick with receivers
              early and add backs when the window opens.
            </p>
            <p>
              In Superflex, a rookie quarterback with a clear path to starting can be
              worth a top pick on its own. Format scarcity makes even an average young
              passer more valuable than a good young receiver. If your league-mates
              skip the second-round passer for a flashy receiver, that is usually your
              edge. See
              <a href="/guides/superflex-vs-1qb">Superflex vs 1QB</a>.
            </p>
            <p>
              Tight ends are a patience position: most prospects take two years to
              become weekly starters. Fine in a rebuild, painful on a contender. If
              your league uses TE premium, the calculus changes; see
              <a href="/guides/te-premium-leagues">TE premium leagues</a>.
            </p>
            <h2 class="static-section-title">Draft by tiers, not by names</h2>
            <p>
              Organize your draft board into tiers, not a ranked list of names. Within
              a tier, players are roughly interchangeable; the edges come from taking
              the last player of a tier instead of the first player of the next one.
              If four receivers sit in your top tier and three are gone, take the
              fourth before you pivot to the top running back of the next tier.
              Reaching across a tier boundary for positional need is how you turn pick
              1.06 into pick 2.02 value.
            </p>
            <p>
              Tiers also tell you when to trade down. If your pick is the last
              selection of a tier and the next three picks all start the next tier,
              moving down a spot or two for a future third costs you nothing in
              expected value. Pay to move up only when it secures the last player of
              a tier, never to grab the first of the next one.
            </p>
            <p>
              Build the tiers before draft day from draft capital, landing spot, and
              your format&rsquo;s scoring. The managers who draft well are better at
              knowing where the cliffs are, not at ranking prospects 1 through 36.
            </p>
            <h2 class="static-section-title">When to trade the pick instead of making it</h2>
            <p>
              A rookie pick is currency, and the draft is not always the best place
              to spend it. The pick&rsquo;s value peaks in the weeks before your
              league&rsquo;s rookie draft, when every manager is dreaming on the same
              highlight reels. That is when you sell if you are not in love with the
              tier available at your slot.
            </p>
            <p>
              Rebuilders should usually keep early picks and trade late ones: the top
              of the draft is where the hit rate justifies the cost, while
              third-round picks are better packaged as sweeteners. Contenders should
              do the opposite: turn mid and late picks into proven production before
              the draft, when pick fever inflates their price.
            </p>
            <p>
              If someone offers a proven 24-year-old starter for your mid first, the
              answer is usually yes. Every pick you keep is a trade you did not make.
              Run the package through the <a href="/trade">trade calculator</a> first,
              then through the
              <a href="/guides/evaluating-a-trade">trade evaluation process</a>.
            </p>
            <h2 class="static-section-title">Common rookie-draft mistakes</h2>
            <ul style="margin-left:20px; line-height:1.8;">
              <li><strong>Drafting the jersey, not the role.</strong> Name recognition
                  from college broadcasts is not a projection. Check the depth chart,
                  the target competition, and the path to snaps before the brand.</li>
              <li><strong>Reaching for running backs in a rebuild.</strong> Draft
                  receivers early, add backs when the window opens.</li>
              <li><strong>Ignoring Superflex QB paths.</strong> A second-round passer
                  who can start in year two is often the best pick on the board and
                  the one your league-mates skip for a flashy receiver.</li>
              <li><strong>Trading future firsts in a panic.</strong> Next year&rsquo;s
                  first is an option on a player who does not exist yet. Selling it to
                  patch a one-week hole is how contenders quietly become average.</li>
              <li><strong>Falling in love with your picks.</strong> The draft is a
                  means, not an identity. If the market offers a proven young starter
                  for your pick, take the proven asset.</li>
              <li><strong>Drafting for need in round one.</strong> Take the best tier
                  available early; fill needs with trades and later picks.</li>
            </ul>
            <div class="highlight-box">
              Draft talent and opportunity, not name recognition. The best rookie picks
              are the ones your league mates aren&rsquo;t talking about yet.
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
              Dynasty trade value is not static, it moves constantly with injuries,
              depth chart changes, hot streaks, and slumps. The managers who win their
              leagues over time are the ones who trade <em>against</em> these
              short-term swings: buying players the market has soured on and selling
              players it has temporarily overrated.
            </p>
            <p>
              Every week, a few players&rsquo; prices detach from their real outlook,
              in both directions. Your job is a process for telling real opportunities
              from traps before you send the offer.
            </p>
            <h2 class="static-section-title">Why the market swings: the overreaction cycle</h2>
            <p>
              Fantasy managers overweight what they just watched. A two-touchdown game
              feels like a breakout; a two-catch game feels like a collapse. Neither is
              usually true after one week, but leagues price both as if they were. This
              recency bias is structural: your league-mates set prices from memory, and
              memory is just the last few box scores.
            </p>
            <p>
              The cycle has a shape. A spike creates believers who bid the price up
              until production regresses and it drifts back. A slump works in reverse:
              panic sets in fast, the price bottoms quickly, and the patient buyer gets
              the discount. You make money by entering at the edges of other
              people&rsquo;s emotions.
            </p>
            <p>
              None of this works without a baseline for what a player is actually
              worth. Anchor every judgment to the
              <a href="/dynasty-trade-value-chart">dynasty trade value chart</a> and
              the <a href="/rankings/dynasty">current rankings</a> before you decide a
              price has moved. Cheap and expensive are meaningless without a reference
              point.
            </p>
            <h2 class="static-section-title">Buy low: the three windows that actually work</h2>
            <p>
              The buy-low that actually works is boring. You are not hunting a player
              who &ldquo;looked bad.&rdquo; You are hunting a player whose
              <em>role</em> is intact while his <em>results</em> are ugly.
            </p>
            <ul style="margin-left:20px; line-height:1.8;">
              <li><strong>The usage is intact, the points are not.</strong> Snaps,
                  target share, and red-zone looks are steady, but touchdowns and
                  yardage have dried up for two or three weeks. Points follow
                  opportunity far more often than opportunity follows points, which
                  makes this the highest-probability buy in fantasy. Confirm the role
                  with <a href="/guides/reading-advanced-metrics">opportunity
                  metrics</a> before you offer.</li>
              <li><strong>The talent is blocked, not broken.</strong> A young player
                  stuck behind an aging or injury-prone starter will eventually get
                  the job, and his price reflects the depth chart rather than his
                  ability. You are buying a future role at a backup&rsquo;s price, so
                  keep the cost proportional to the real risk that he stays
                  blocked.</li>
              <li><strong>The injury discount.</strong> A player coming off a minor
                  injury, where the panic is bigger than the long-term risk. Markets
                  treat all injuries as equal for about two weeks, so a three-game
                  soft-tissue absence gets nearly the same discount as a structural
                  problem. Know which one you are buying.</li>
            </ul>
            <p>
              In each case, the test is the same: would you still want the player if
              the slump lasted two more weeks? If yes, the price is the opportunity.
              If no, you are talking yourself into a bargain that is not one.
            </p>
            <h2 class="static-section-title">Sell high: the three windows that actually work</h2>
            <p>
              Selling high is socially harder than buying low: your league-mates just
              watched the player eat. You do not need the absolute top. You need to
              move a declining or lucky profile into a younger or stickier one before
              the market catches up.
            </p>
            <ul style="margin-left:20px; line-height:1.8;">
              <li><strong>The touchdown mirage.</strong> A player riding a touchdown
                  rate his opportunity cannot support. Five touchdowns on a 12% target
                  share is a loan from variance, and variance collects. Sell while the
                  box score is doing your marketing.</li>
              <li><strong>The aging veteran&rsquo;s last great month.</strong> An older
                  running back coming off a big stretch: sell the name before the
                  cliff. The market pays for recent production; the curve says the role
                  is living on borrowed time. A contender&rsquo;s fair rental price
                  beats what he will be worth to you in six months.</li>
              <li><strong>The fill-in spike.</strong> A backup who spiked in value
                  during a short injury fill-in for a starter who is about to return.
                  The role evaporates the moment the starter is back, and so does the
                  price. Move him the week before the return, not the week after.</li>
            </ul>
            <p>
              A good sell is one you still like a little; if you are desperate to dump
              the player, you waited too long.
            </p>
            <h2 class="static-section-title">Confirming with data instead of vibes</h2>
            <p>
              The clearest buy-low and sell-high signals show up as movement in value
              over time. The <a href="/top-movers">top movers</a> page tracks which
              players are rising and falling fastest, and
              <a href="/trade-intel">trade intelligence</a> surfaces market signals
              from real league activity. Pair those with the
              <a href="/rankings/dynasty">current rankings</a> to spot gaps between a
              player&rsquo;s price and his true outlook.
            </p>
            <p>
              Then verify the story underneath the movement. A falling price with intact
              usage is a buy signal; a falling price with collapsing snaps is just the
              market being right. A rising price on growing target share is a
              breakout; a rising price on touchdowns alone is a sell window. Make the
              movers list a weekly habit, same day every week: the managers who catch
              every window are not smarter, they just look regularly.
            </p>
            <h2 class="static-section-title">Timeline fit: the same dip, two different answers</h2>
            <p>
              Then check the deal against your timeline. A buy-low running back is a
              gift to a <a href="/guides/contending-in-dynasty">contender</a> and a
              trap to a <a href="/guides/dynasty-rebuild-strategy">rebuilder</a> who
              just added another 27-year-old. The same dip can be a win or a mistake
              depending on whether you need 2026 points or 2028 optionality.
            </p>
            <p>
              This is the step most managers skip. A 28-year-old receiver at a 20%
              discount is a great buy-low in the abstract and a terrible one for a
              team whose window opens in two years, because the discount expires before
              the window does. Always ask: will this player still be good when I am
              good? Rebuilders buy youth dips; contenders buy production dips.
              Different timelines, different shopping lists, same discipline.
            </p>
            <h2 class="static-section-title">Mistakes that turn patience into losses</h2>
            <ul style="margin-left:20px; line-height:1.8;">
              <li><strong>Forced contrarianism.</strong> Some players are simply good,
                  and the market is correctly expensive. If the metrics, the age curve,
                  and the depth chart all agree with the price, leave the player
                  alone.</li>
              <li><strong>Buying the dip on a lost role.</strong> A falling price with
                  falling snaps is not a discount, it is information. The buy-low
                  requires intact usage. Without it, you are catching a falling knife
                  and calling it value.</li>
              <li><strong>Bidding against yourself.</strong> Opening with your best
                  offer on a buy-low target tells the seller exactly how much you want
                  the player. Start fair but leave room; the whole point of a dip is
                  that you should not have to pay full price.</li>
              <li><strong>Selling a core piece for the thrill of it.</strong>
                  Sell-high applies to lucky profiles and aging rentals, not to
                  24-year-old target earners having a great month. Do not get so
                  addicted to selling spikes that you trade away the players spikes
                  are made of.</li>
              <li><strong>Confusing a buy-low with a lottery ticket.</strong> A young
                  player with no role and no production is not buying low, he is just
                  cheap. There has to be a reason the price recovers: talent, draft
                  capital, a path to snaps. Hope is not a catalyst.</li>
            </ul>
            <div class="highlight-box">
              The market overreacts to recent results. Your edge is patience: buy the
              dip on talent, sell the spike on age and luck.
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
              Adding up trade values on each side of a deal is a useful first check, but
              it is only the beginning. The best trades aren&rsquo;t always the ones that
              &ldquo;win&rdquo; on raw value, they&rsquo;re the ones that make
              <em>your</em> roster better for <em>your</em> timeline. Here&rsquo;s a
              repeatable process.
            </p>
            <p>
              Run every significant deal through these steps in order. Each one catches
              a different kind of mistake, and skipping ahead is how fair-looking trades
              quietly lose leagues.
            </p>
            <h2 class="static-section-title">Step 1: Check the raw value</h2>
            <p>
              Start by comparing the total value on each side using format-appropriate
              numbers (1QB or Superflex). A quick way to do this is the
              <a href="/trade">trade calculator</a>, which grades both sides and
              suggests counters. If a deal is wildly lopsided on value, you usually
              have your answer.
            </p>
            <p>
              Use the same settings your league actually plays. Superflex vs 1QB is the
              big one (see <a href="/guides/superflex-vs-1qb">Superflex vs 1QB</a>), but
              TE premium and roster size also move the result. A deal that looks fair on
              a generic chart can be a steal or a disaster once those knobs match your
              league.
            </p>
            <p>
              Sanity-check the calculator against the
              <a href="/dynasty-trade-value-chart">dynasty trade value chart</a> for any
              single player who dominates the deal. Calculators aggregate well but can
              blur one cornerstone piece. If one player is most of a side&rsquo;s
              value, price him individually first.
            </p>
            <h2 class="static-section-title">Step 2: Account for consolidation</h2>
            <p>
              Two good players are generally worth more than three mediocre ones,
              because starting lineup spots are limited and the best players are the
              hardest to replace. When you trade multiple pieces for one stud, expect,
              and accept, paying a small value premium for that consolidation.
            </p>
            <p>
              The reverse is true in a rebuild: unpacking a star into several younger
              pieces can be correct even if you &ldquo;lose&rdquo; a few points of
              value, because you cannot start the star enough times to justify holding
              him through a two-year trough. That is a timeline decision, not a
              calculator error.
            </p>
            <p>
              The test: after the trade, count how many weekly starters you have, not
              how many total points of value. If the deal adds a starter and costs you
              two bench players, you won the consolidation math regardless of what the
              grade says.
            </p>
            <h2 class="static-section-title">Step 3: Match the deal to your timeline</h2>
            <p>
              Are you contending or rebuilding? Contenders should trade youth and picks
              for proven, win-now production. Rebuilders should do the reverse: sell
              aging stars for young players and draft capital. A trade that&rsquo;s
              &ldquo;fair&rdquo; on value can still be wrong if it doesn&rsquo;t fit
              where your team is in its cycle.
            </p>
            <p>
              If you are unsure which mode you are in, count startable weeks and
              upcoming draft capital, not last year&rsquo;s record. A 5-9 team with
              three firsts is a rebuild even if the chat still thinks you are &ldquo;a
              quarterback away.&rdquo; Walk through
              <a href="/guides/dynasty-rebuild-strategy">rebuild strategy</a> or
              <a href="/guides/contending-in-dynasty">contending strategy</a> before
              you accept a deal that fights your window.
            </p>
            <p>
              Be honest about the middle. Most teams are neither pure contenders nor
              pure rebuilders, and the correct move is usually to pick a direction
              rather than straddle. A fair trade that keeps you in sixth place for
              three years is the most expensive kind.
            </p>
            <h2 class="static-section-title">Step 4: Value positional scarcity and need</h2>
            <p>
              A player is worth more to a roster that needs his position. Don&rsquo;t
              trade from a position of strength into another position of strength,
              address real lineup holes. In Superflex, weigh quarterback depth
              especially heavily: the gap between QB12 and QB24 is a season.
            </p>
            <p>
              Scarcity is league-specific. In a 14-team Superflex league, a starting
              quarterback is close to irreplaceable; in a 10-team 1QB league, you can
              stream the position. Price the player against your league&rsquo;s
              replacement level, not the national consensus. The
              <a href="/compare">compare tool</a> is useful here: line up your starter
              against the best available alternative and look at the actual gap.
            </p>
            <p>
              Need cuts both ways at the negotiating table. Never let the other
              manager know which position you are desperate to fill until the price is
              set. Desperation is a tax you pay voluntarily.
            </p>
            <h2 class="static-section-title">Step 5: Look past this week</h2>
            <p>
              Before you finalize, sanity-check the underlying trends from the
              <a href="/guides/reading-advanced-metrics">advanced metrics</a> and the
              <a href="/top-movers">top movers</a> page. You want to be buying
              ascending players and selling declining ones, not the reverse. A player
              with three straight weeks of growing target share is a different asset
              from one with the same season total built on a single outlier game.
            </p>
            <p>
              Check the age curve too. A 24-year-old WR2 coming off back-to-back
              top-20 finishes is appreciating; a 28-year-old RB with one year left of
              elite production is depreciating. The calculator prices the present well.
              Your job is to price the direction. See
              <a href="/guides/positional-aging-curves">positional aging curves</a>.
            </p>
            <p>
              Then sleep on any deal that moves a cornerstone. Dynasty trades are
              rarely so urgent that you must accept before the next snap. If the other
              manager is rushing you, that is information too.
            </p>
            <h2 class="static-section-title">Step 6: Negotiate the deal, not just the value</h2>
            <p>
              Most rejected trades fail on framing, not math. Lead with what the other
              manager gets, not what you want. &ldquo;This gives you two starters for
              your playoff push&rdquo; closes more deals than &ldquo;I need a running
              back.&rdquo; People accept trades they can defend in the group chat.
            </p>
            <p>
              Counter instead of rejecting. A first offer is an opening position, and a
              flat no ends a negotiation that a counter would have continued. If the
              calculator shows a gap, close it with the smallest piece that works: add
              a pick, swap a bench player, adjust a pick round. Small moves signal
              reasonableness and keep the conversation alive.
            </p>
            <p>
              Know your walk-away number before you start talking. Decide the least
              you would accept or the most you would pay, and do not move it because
              the chat got exciting. The managers who lose trades are usually the ones
              who negotiated against themselves.
            </p>
            <h2 class="static-section-title">Worked example: putting it all together</h2>
            <p>
              Say you are a contender, and you are offered a 27-year-old running back
              having a career year for your 23-year-old WR3 plus a second-round pick.
              Step 1: the calculator calls it close, maybe even slightly in your
              favor. Do not stop there.
            </p>
            <p>
              Step 2: you are consolidating two lesser assets into one starter, which
              favors the deal. Step 3: the timeline fits, since you need points now
              and the pick is a future asset you can spend. Step 4: check your running
              back room. If he becomes your RB1, the need is real; if he is your RB3,
              you are paying a starter&rsquo;s price for depth.
            </p>
            <p>
              Step 5 is where this deal usually gets done or dies. Is his production
              backed by a huge snap share, or by touchdowns? A 27-year-old back is a
              rental, so are you paying a rental price or a cornerstone price? If the
              usage is real and the price reflects one to two seasons of production,
              accept. If you are paying three seasons of value for one season of age,
              counter down or walk away.
            </p>
            <div class="highlight-box">
              A good trade makes your starting lineup better for your timeline. Value
              is the starting point; fit is the decision.
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
              production for future optionality. Done well, a rebuild lasts one
              offseason and one ugly year. Done poorly, it becomes a five-year hobby of
              collecting &ldquo;upside&rdquo; that never starts.
            </p>
            <h2 class="static-section-title">Admit the window is closed</h2>
            <p>
              Look at age, picks, and starting lineup quality together. If your
              skill-position core is 27-plus, you have no first-round picks in the next
              two drafts, and you are already out of the playoff picture by week
              eight, you are not &ldquo;a running back away.&rdquo; You are a seller.
              Staying in the middle, good enough to finish 7-7, is the most expensive
              place in dynasty. Mediocre teams draft late, lose trade leverage, and
              age out slowly.
            </p>
            <p>
              The first seller to the market gets the best prices; the fourth seller
              gets sympathy.
            </p>
            <h2 class="static-section-title">Sell the right aging pieces, and sell them early</h2>
            <p>
              Your best trade chips are productive veterans that contenders need this
              season: aging running backs, proven receivers still posting target
              share, and (in Superflex) a quarterback you cannot wait on. Price them
              with <a href="/guides/dynasty-trade-value">dynasty trade values</a>,
              then prefer packages that return young players with roles plus draft
              capital, not just a pile of thirds.
            </p>
            <p>
              Timing is half the trade. A 28-year-old running back is worth a
              first-round pick in October to a team chasing a title. The same player
              is worth a second in February after a lost season of wear. Sell
              production while it still looks like production. This is where the
              <a href="/guides/positional-aging-curves">positional aging curves</a>
              matter most: running backs depreciate fastest, receivers hold value
              longer, and quarterbacks barely depreciate at all. A rebuilding team
              should be selling the first category, holding the third, and being
              selective about the second.
            </p>
            <p>
              Do not sell young players with sticky usage just because you are
              rebuilding. A 23-year-old WR2 with 20% target share <em>is</em> the
              rebuild. Selling him for a future first feels active and is often a
              downgrade. The rule is simple: sell players whose best seasons are
              behind them, keep players whose best seasons are ahead, and be honest
              about which is which.
            </p>
            <h2 class="static-section-title">What you should be collecting</h2>
            <ul style="margin-left:20px; line-height:1.8;">
              <li><strong>Rookie firsts and early seconds</strong>, especially in years
                  with a known quarterback or receiver class. See
                  <a href="/guides/rookie-draft-strategy">rookie draft strategy</a>.
                  Future firsts from likely bad teams are the best currency in
                  dynasty; a first projected early can be worth double a first
                  projected late.</li>
              <li><strong>Young players whose opportunity is rising</strong> before the
                  points show up, the same profiles the
                  <a href="/breakouts">breakout engine</a> is built to flag. A
                  24-year-old receiver coming off back-to-back top-20 finishes with
                  growing target share is worth more than a rookie pick that has never
                  played a snap.</li>
              <li><strong>Quarterback youth in Superflex</strong>, even if they are
                  sitting this year. Rebuilds that ignore passers in Superflex restart
                  in three years. See
                  <a href="/guides/superflex-vs-1qb">Superflex vs 1QB</a> for why the
                  format changes every valuation.</li>
              <li><strong>Tight end youth with a pulse</strong> in any format, and
                  especially in
                  <a href="/guides/te-premium-leagues">TE premium leagues</a>. The
                  position takes two to three years to develop, which is exactly the
                  length of your rebuild window.</li>
            </ul>
            <h2 class="static-section-title">The rebuild timeline: one ugly year, not five</h2>
            <p>
              A healthy rebuild has a shape. The first offseason is for selling: you
              move aging production for picks and young roles before values slide.
              The first season is for losing on purpose-adjacent terms: your lineup is
              young, your record is bad, and your draft position improves with every
              loss. The second offseason is for turning surplus into starters:
              packaging picks for a proven young player, not hoarding them forever.
            </p>
            <p>
              If you still cannot name a future starting lineup after two rookie
              drafts, you either sold the wrong players or you kept drafting running
              backs. A healthy rebuild produces a competitive roster as soon as the
              young core&rsquo;s usage arrives, not when every pick has
              &ldquo;hit.&rdquo; Switch from collecting to
              <a href="/guides/contending-in-dynasty">contending</a> the moment your
              starting lineup can actually win weeks. Holding picks past that point is
              how rebuilders miss their window on the way up.
            </p>
            <h2 class="static-section-title">How rebuilders should spend draft picks</h2>
            <p>
              Rookie picks are the most overvalued and most misused asset in dynasty.
              The miss rate on even first-round rookie picks is real, which means
              picks are often better spent as currency than as lottery tickets. That
              does not mean never draft. It means draft with a plan instead of
              drafting whoever falls.
            </p>
            <p>
              Early in a rebuild, lean toward drafting and stashing: your roster has
              room, your timeline is long, and a rookie who needs a year is fine. Late
              in a rebuild, when the core is nearly set, flip the surplus. Two
              mid-firsts for a proven 24-year-old starter is the classic window-opening
              move. Rebuilders who draft six rookies a year for three straight years
              end up cutting second-year players to make room for new darts. That is
              not asset accumulation. That is churn.
            </p>
            <h2 class="static-section-title">Trades you should not take just to &ldquo;get younger&rdquo;</h2>
            <p>
              Youth is not automatically good. A 22-year-old with 8% snap share and a
              crowded depth chart is not a building block; he is a dart. A 26-year-old
              receiver with 22% target share may be the best remaining core piece you
              have. Rebuilds fail when every productive veteran is sold for a pick
              that will be used on a running back two years from now.
            </p>
            <p>
              Use the <a href="/trade">trade calculator</a> to keep yourself honest,
              then apply the same
              <a href="/guides/evaluating-a-trade">fit checks</a> you would in any
              other deal. If the package does not leave you with startable youth or
              premium picks, you did not rebuild. You just got worse.
            </p>
            <h2 class="static-section-title">Common rebuild mistakes, answered directly</h2>
            <ul style="margin-left:20px; line-height:1.8;">
              <li><strong>&ldquo;Should I tank for a better pick?&rdquo;</strong>
                  Deliberately setting bad lineups is against the spirit of most
                  leagues and often against the rules. You do not need to tank. A
                  young rebuilding roster loses naturally. Keep setting your best
                  lineup; the losses will come.</li>
              <li><strong>&ldquo;When do I stop selling?&rdquo;</strong> When your
                  remaining veterans are either part of the next window or worth more
                  to your lineup than the market will pay. Selling a productive
                  26-year-old for 60 cents on the dollar is worse than keeping
                  him.</li>
              <li><strong>&ldquo;What if nobody is buying?&rdquo;</strong> Then you
                  priced too high or you waited too long. Drop the ask, target the two
                  or three teams with real title odds, and remember that a good offer
                  in October beats a great offer that never comes.</li>
            </ul>
            <div class="highlight-box">
              Tear down once, sell aging production for youth and picks, and stop
              rebuilding the week your young core can win games.
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
              Contending is the opposite of collecting. You already have a starting
              lineup that can win a title this season, so every trade should be judged
              by whether it improves the weeks you will actually play, especially the
              playoff slate, not by whether it makes your roster &ldquo;younger&rdquo;
              on paper.
            </p>
            <h2 class="static-section-title">Confirm you are actually a contender</h2>
            <p>
              Record is a lagging indicator. Look at starting-lineup value, remaining
              schedule, and whether your core still has a year of production left. A
              6-3 team of 29-year-old running backs can be a contender this year and a
              rebuild in March. A 4-5 team with elite young skill players and a soft
              playoff schedule can still be a buyer.
            </p>
            <p>
              Run a simple audit. List your weekly starters and mark each one as
              &ldquo;bankable,&rdquo; &ldquo;fine,&rdquo; or &ldquo;hope.&rdquo; If
              more than two starters are hope, you are not a contender; you are a wish.
              Then check the playoff format: in a six-team bracket with byes, seeding
              matters enormously, and a team sitting third may need different moves
              than a team fighting for the last spot. Your record tells you where you
              are. Your starters tell you where you are going.
            </p>
            <p>
              Use the <a href="/rankings/dynasty">rankings</a> and your league&rsquo;s
              playoff-odds tools to sanity-check the chat. If you are a true bubble
              team, small buys (a RB2, a QB2 in Superflex) beat blockbuster sells of
              future firsts. The goal is to turn a coin-flip team into a favorite, not
              to turn a favorite into a coin flip.
            </p>
            <h2 class="static-section-title">Know your window&rsquo;s shape and expiry date</h2>
            <p>
              Not all contention windows look the same. A veteran window is built on
              aging production: backs and receivers in their late twenties who will
              fall off a cliff together, usually within one to two seasons. This
              window demands urgency. A young-core window is built on players 25 and
              under who are already producing; it can stay open for years and should
              be extended carefully rather than mortgaged. A one-year window appears
              when a rival&rsquo;s injury or a surprise breakout temporarily clears
              your path; it is real, but you should not pay three-year prices for it.
            </p>
            <p>
              The <a href="/guides/positional-aging-curves">positional aging curves</a>
              tell you which windows close fastest. Running-back-heavy contenders have
              the shortest fuse: when the cliff comes, it comes for the whole position
              group at once. Receiver- and quarterback-based contenders age more
              gracefully and can afford to be patient at the deadline. Name your window
              type before the trade deadline, because each type buys differently.
            </p>
            <h2 class="static-section-title">What contenders should buy</h2>
            <ul style="margin-left:20px; line-height:1.8;">
              <li><strong>Reliable weekly production</strong> at positions you actually
                  start. Volume running backs and high-floor receivers beat dart-throw
                  rookies in November. A 28-year-old back with one year of elite
                  production left is not a dynasty asset. He is a title asset, and
                  that is exactly what you are shopping for.</li>
              <li><strong>Injury insurance</strong> at your thinnest position. One
                  hamstring should not turn a title favorite into a streamer. Back up
                  the position where a single injury would cost you the most expected
                  points, not the position where you are already deep.</li>
              <li><strong>Quarterback depth in Superflex.</strong> Format scarcity
                  makes a startable QB2 more valuable in December than another young
                  WR4. See <a href="/guides/superflex-vs-1qb">Superflex vs 1QB</a>. If
                  your second quarterback slot is a weekly question mark, that is your
                  biggest leak.</li>
              <li><strong>Playoff-schedule upgrades.</strong> A player with soft
                  matchups in your league&rsquo;s semifinal and final weeks is worth a
                  small premium over an equal player facing elite defenses. Marginal
                  edges decide close playoff games.</li>
            </ul>
            <h2 class="static-section-title">How to price a rental without wrecking the future</h2>
            <p>
              The right cost is usually a future first plus a depth piece, or a young
              player who is not in your starting lineup. Price it with the
              <a href="/trade">trade calculator</a> and the process in
              <a href="/guides/evaluating-a-trade">how to evaluate a trade</a>.
              Paying a premium is expected. Paying two future firsts for a rental who
              does not start for you is how contenders accidentally rebuild.
            </p>
            <p>
              The key distinction is whether the player starts for you. A rental who
              moves into your lineup every week can justify a first-round pick,
              because you are buying 8 to 10 starts that matter. A rental who sits
              behind your starters is a luxury, and luxuries should cost thirds, not
              firsts. Before you offer, write down the player&rsquo;s role on your
              team in one sentence. If the sentence includes the word
              &ldquo;depth,&rdquo; lower the bid.
            </p>
            <p>
              Also price the alternative: what does the same pick buy you in the
              offseason? A future first spent in October on a 29-year-old back is
              gone. A future first kept until February might buy a 24-year-old starter
              from a rebuilder. Only pay the October price when the title odds
              genuinely move.
            </p>
            <h2 class="static-section-title">In-season vs offseason buys</h2>
            <p>
              In-season, prioritize players with roles <em>now</em>. Offseason, you can
              still contend and improve the long-term core at the same time,
              especially at receiver and quarterback, where
              <a href="/guides/positional-aging-curves">aging curves</a> are slower.
              The offseason is also when rebuilders overpay for your aging backs. That
              is the <a href="/guides/buy-low-sell-high">sell-high</a> window you
              should use if a veteran just carried you through a title run.
            </p>
            <h2 class="static-section-title">Lineup weeks, not trophy case</h2>
            <p>
              Contending is about the weeks you will start a player, not about
              collecting every name that might be good in 2028. If a young receiver is
              your WR5, he is a trade chip for a back who will start in December. If
              your Superflex QB2 is a streamer, fix that before you add another
              prospect.
            </p>
            <p>
              Check remaining schedule and injury risk the same way you would in
              redraft. Dynasty value still matters so you do not torch the future for
              a two-week rental, but a title is worth a future late first. It is
              rarely worth two early firsts and your cheapest young starter.
            </p>
            <h2 class="static-section-title">Mistakes that end windows early</h2>
            <ul style="margin-left:20px; line-height:1.8;">
              <li><strong>Trading a cornerstone in his prime</strong> to &ldquo;get
                  picks back&rdquo; during a year you can win. Picks are for the
                  future. This year is the present.</li>
              <li><strong>Buying aging running backs in August</strong> when your
                  window is actually next season. August prices assume this-year
                  production; if you do not need it yet, wait for the in-season
                  discount.</li>
              <li><strong>Emptying the taxi squad</strong> of every interesting young
                  player for a committee back who will be irrelevant in 14 months,
                  unless that back is the difference in a title week.</li>
              <li><strong>Ignoring the second quarterback slot</strong> in Superflex
                  while adding a fourth receiver. Fix the scarcest position
                  first.</li>
            </ul>
            <div class="highlight-box">
              If the lineup can win the league, buy production that starts. Save the
              youth movement for the winter after a real window closes.
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
              Tight-end premium (TEP) leagues award extra points per tight-end
              reception on top of whatever PPR the rest of the roster uses. The usual
              bump is +0.5 PPR for tight ends. That sounds small. Over a 17-game season it is
              not. It stretches the gap between the few tight ends who earn real
              volume and everyone else, and it changes how you should spend draft
              capital and trade chips.
            </p>
            <h2 class="static-section-title">Why the elite tier gets expensive</h2>
            <p>
              In standard PPR, a tight end competing with receivers for targets is
              often a positional afterthought. In TEP, those same targets are worth
              more than a receiver&rsquo;s. The handful of tight ends who see 20% or
              more of their team&rsquo;s targets become weekly difference-makers, not
              just &ldquo;TE1s.&rdquo; The market responds by pushing them up
              <a href="/rankings/dynasty">dynasty rankings</a> relative to a
              non-premium chart.
            </p>
            <p>
              That does not mean every tight end is a buy. The position is still
              bimodal: a small premium tier, a wide middle that is startable but not
              special, and a long tail of streaming options. Paying a first-round
              startup pick for a mid-tier tight end because &ldquo;it&rsquo;s
              TEP&rdquo; is how you strand capital. The premium rewards volume, not
              the position label. A tight end with 5 targets a game gets almost
              nothing from TEP; one with 8 targets a game gets a real boost. Pay for
              the targets, not the roster designation.
            </p>
            <h2 class="static-section-title">Startup draft effects</h2>
            <p>
              In a TEP startup, it is reasonable to take a true difference-making
              tight end earlier than you would in a 1-PPR league, especially if you
              are already set at quarterback in Superflex. A 24-year-old every-down
              tight end is a positional advantage that lasts for years, and TEP is the
              one scoring setting that makes the position worth an early pick.
            </p>
            <p>
              It is not reasonable to take the TE8 over a young receiver with a clear
              role. The middle of the position is still the middle: startable,
              replaceable, and available later. Use
              <a href="/guides/startup-draft-guide">startup draft strategy</a> and
              check values on the
              <a href="/dynasty-trade-value-chart">trade value chart</a> rather than
              following a generic &ldquo;TEP cheat sheet&rdquo; that ignores the rest
              of your build.
            </p>
            <p>
              The practical rule: pay the premium for the top tier, draft the middle
              tier at a discount, and stream the rest. The managers who win TEP
              startups are the ones who buy one difference-maker and let everyone
              else fight over TE9 through TE16.
            </p>
            <h2 class="static-section-title">Rookie drafts and development timelines</h2>
            <p>
              In rookie drafts, TEP raises the floor of early-declare tight ends with
              draft capital, but it does not erase development time. Most tight ends
              still take a year or two. A first-round NFL tight end is a better bet in
              TEP than in standard, but he is still a year-two asset, not a year-one
              starter.
            </p>
            <p>
              Rebuilders can wait; the timeline fits. Draft the young tight end, stash
              him, and let the premium compound. Contenders should prefer a proven
              volume tight end over a prospect unless the prospect is a clear premium
              talent. A contender spending a first-round rookie pick on a tight end
              who will not start this year is spending a win-now asset on a win-later
              player. See
              <a href="/guides/rookie-draft-strategy">rookie draft strategy</a>.
            </p>
            <h2 class="static-section-title">Trading in TEP</h2>
            <p>
              When you evaluate a deal, make sure both sides are scored as TEP. A
              tight end who looks &ldquo;expensive&rdquo; on a standard chart is often
              fairly priced once the premium is applied. The
              <a href="/trade">trade calculator</a> is the place to toggle that, then
              apply the same
              <a href="/guides/evaluating-a-trade">fit checks</a> you would for any
              other position: need, age, and whether you are contending.
            </p>
            <p>
              TEP creates a specific trade market: managers without a premium tight
              end will overpay at the deadline when the streaming options dry up. If
              you hold two, that is leverage. Shop the second one to the desperate
              team in October, not to the comfortable team in August. And when you are
              buying, buy in the offseason, when nobody is thinking about positional
              scarcity.
            </p>
            <p>
              Holding two premium tight ends is a real strategy in TEP because the
              waiver replacements are so much worse than at receiver. Holding four is
              usually a traffic jam. Trade the third for a need, do not wait for a
              perfect offer that never comes.
            </p>
            <h2 class="static-section-title">Roster construction: how many tight ends is enough</h2>
            <p>
              Two startable tight ends is the sweet spot in most TEP leagues: a weekly
              starter plus either a second premium option for flex or a young upside
              stash. Three is defensible if the third is a developing prospect with
              draft capital. Four is a roster clog that costs you depth at positions
              with more lineup spots.
            </p>
            <p>
              The waiver wire math explains why. In TEP, the gap between TE6 and TE18
              is enormous, while the gap between WR30 and WR50 is small. That means
              tight end depth has real trade value and receiver depth does not. Carry
              the extra tight end over the extra receiver when the choice is close,
              because the tight end is the one you can actually trade.
            </p>
            <p>
              In leagues with a dedicated tight end flex spot, add one more to the
              target. The extra starting slot doubles the value of depth at the
              position.
            </p>
            <h2 class="static-section-title">Mistakes that waste the premium</h2>
            <ul style="margin-left:20px; line-height:1.8;">
              <li><strong>Paying premium prices for non-premium volume.</strong> The
                  extra half point only matters if the catches exist. A blocking-first
                  tight end with 40 catches gains 20 points on the season. That is not
                  worth a first-round pick.</li>
              <li><strong>Drafting TE2s early in startups.</strong> The middle tier is
                  deep and cheap. Spend early capital on the difference-makers or on
                  other positions, then take your TE2 in the middle rounds.</li>
              <li><strong>Forgetting to toggle TEP in trade talks.</strong> Both sides
                  of every negotiation must use TEP values. A deal that looks fair on
                  standard scoring is a discount for the team receiving the tight
                  end.</li>
              <li><strong>Stashing four tight ends and no receivers.</strong> Depth has
                  to convert to lineup spots. Trade the surplus before it rots on
                  your bench.</li>
              <li><strong>Chasing last year&rsquo;s touchdowns.</strong> TEP rewards
                  targets, not scores. A tight end with 90 targets and 4 touchdowns is
                  a better TEP asset than one with 60 targets and 9 touchdowns, and
                  he is usually cheaper.</li>
            </ul>
            <div class="highlight-box">
              TEP makes true volume tight ends worth paying up for. It does not make
              every tight end a first-round asset.
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
              A dynasty startup is the one draft where the entire player pool is
              available at once. That makes it the highest-leverage event in the format
              and the easiest place to copy a flashy board instead of building a team.
              The goal is not to draft the most famous names. It is to leave with a
              starting lineup that has a timeline and enough depth to survive the first
              injury wave.
            </p>
            <h2 class="static-section-title">Pick a window before pick 1.01</h2>
            <p>
              Decide whether this startup is a win-now build, a balanced core, or a
              youth pile. You can adjust later, but the first four rounds should not
              fight each other. Mixing a 30-year-old bell-cow, two rookie receivers,
              and no quarterback plan in Superflex is how startups become immediate
              rebuilds.
            </p>
            <p>
              If you want to contend early, bias toward proven volume and accept that
              some of those players will need to be sold in two years. You are renting
              the present and you know it. If you want to grow, bias toward young
              receivers and (in Superflex) young passers, and be willing to be bad in
              year one. Both are valid.
              <a href="/guides/contending-in-dynasty">Contending</a> and
              <a href="/guides/dynasty-rebuild-strategy">rebuilding</a> are just those
              plans with the draft already over.
            </p>
            <p>
              The balanced core is the default for most managers and the hardest to
              execute: young enough to last, productive enough to win now. It requires
              discipline in the middle rounds, where the temptation is to draft names
              instead of roles. Whatever you choose, write it down before the draft
              starts. A plan you can check beats a feeling you cannot.
            </p>
            <h2 class="static-section-title">Format first, then position</h2>
            <p>
              In Superflex, elite and good-enough quarterbacks come off the board early
              for a reason. Falling behind at the position in round two is a hole you
              will spend three years filling. The math is simple: with up to 24
              passers starting league-wide, the replacement level is brutal. Draft your
              QB1 early, your QB2 before the run ends, and consider a third passer as
              insurance if the board allows.
            </p>
            <p>
              In 1QB, let someone else take the extra passer and spend the capital on
              skill-position talent. A second quarterback in 1QB is a handcuff, not a
              cornerstone, and every round you spend on one is a round you did not
              spend on a young receiver with a role.
            </p>
            <p>
              Tight-end premium similarly pulls true volume TEs up the board. Everyone
              else should still be ranked by the same
              <a href="/guides/dynasty-trade-value">trade-value</a> logic you will use
              after the draft: age, role, and replacement cost. Details are in
              <a href="/guides/superflex-vs-1qb">Superflex vs 1QB</a>.
            </p>
            <h2 class="static-section-title">ADP is a clock, not a ranking</h2>
            <p>
              Startup ADP tells you when you will have to reach. It does not tell you
              who is better. If a player you want is going two rounds later than his
              value, wait. If the player you need never makes it back, that is the
              cost of nominating a plan. Compare ADP to the
              <a href="/dynasty-trade-value-chart">value chart</a> and to
              <a href="/guides/adp-vs-trade-value">ADP vs trade value</a> so you are
              not drafting a consensus board that already has no edge.
            </p>
            <p>
              One practical use: track which positions are flying off your specific
              board versus national ADP. Every draft room has its own biases. If your
              league overdrafts running backs, the value falls to receivers by round
              4. Adjust in real time instead of drafting the generic plan.
            </p>
            <p>
              The inefficiency you want is a player the room drafts in round 8 who
              your board prices as a round-5 value. That is not a reach. It is the
              whole game.
            </p>
            <h2 class="static-section-title">Rounds 5 through 12 decide the draft</h2>
            <p>
              Rounds 5 through 12 are where startups are decided. This is where
              managers either stack young receivers with roles or panic-draft a
              running back committee. The stars are interchangeable by round 5; the
              edges live in the middle.
            </p>
            <p>
              Prefer players with snaps and targets, even if they are less famous. A
              25-year-old WR3 with a locked-in 20% target share will outscore the
              bigger name who just lost his starting job, and he costs four rounds
              less. Use
              <a href="/guides/reading-advanced-metrics">advanced metrics</a> and
              <a href="/prospects">prospect profiles</a> instead of recency from last
              year&rsquo;s playoffs.
            </p>
            <p>
              Positional runs are the middle-round tax. When six tight ends go in
              eight picks, do not join the run at its peak unless the player is
              genuinely the last of a tier. Let the room overpay for the position and
              take the value that falls. Patience in the middle rounds is a
              competitive advantage because almost nobody has it.
            </p>
            <h2 class="static-section-title">The endgame: upside that does not need a lineup spot</h2>
            <p>
              Late, take upside that does not need a starting spot this year: rookies
              with draft capital, backup passers in Superflex, and injured players
              whose landing spot still makes sense. Your taxi squad and deep bench are
              the right home for darts.
            </p>
            <p>
              Do not take a 28-year-old free agent running back &ldquo;because he
              might get signed.&rdquo; That is redraft thinking, and it wastes the one
              part of the draft where dynasty managers are supposed to have an edge.
              Every endgame pick should answer one question: what does this player
              look like if the best realistic case happens? If the answer is a flex
              starter, pass. If the answer is a multi-year starter, take the swing.
            </p>
            <p>
              Handcuffs are endgame luxury, not strategy. One handcuff for your own
              early running back is fine. Three handcuffs for other people&rsquo;s
              backs is a bench full of lottery tickets that only pay if someone else
              gets hurt.
            </p>
            <h2 class="static-section-title">After the draft: the first 48 hours</h2>
            <p>
              The draft is not the end of team-building; it is the end of the
              beginning. The first two days after a startup are the most active
              trading window your league will ever have, because every roster has
              obvious surpluses and holes.
            </p>
            <p>
              Audit your roster honestly. Count your weekly starters by position and
              find the thinnest spot. Then shop your surplus before week 1: the fourth
              receiver you drafted is worth more to the manager who left the draft
              with two. Package depth into starters where you can, because
              consolidation wins championships. The
              <a href="/trade">trade calculator</a> is useful here, since startup
              values are still close to draft-day prices and both sides can see the
              math.
            </p>
            <p>
              Do not overreact to week 1. Every startup manager who drafts young will
              be tempted to blow it up after one bad Sunday. Your plan was built for a
              season, not a week. Make the trades that fix real holes, then let the
              roster play.
            </p>
            <div class="highlight-box">
              Choose a window, honor the format, and spend the middle rounds on roles,
              not names. The startup is a roster, not a celebrity list.
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
              The waiver wire is the only market in fantasy where you compete against
              the whole league every week without needing someone to say yes. In
              redraft it is how championships are patched together. In dynasty it is
              how you find the cheap young player who becomes next year&rsquo;s trade
              chip. The mechanics differ (priority list vs FAAB budget), but the
              evaluation does not: you are bidding on opportunity, not on last
              Thursday&rsquo;s box score.
            </p>
            <h2 class="static-section-title">FAAB vs waiver priority</h2>
            <p>
              FAAB (Free Agent Acquisition Budget) is a season-long pile of dollars you
              bid in secret. Hitting 100% of your budget in week three on a backup who
              lost the job by week six is a classic mistake. Hitting 0% until December
              and watching every useful add go to more aggressive managers is the
              opposite mistake. A simple split: save a real bid (often 15-40% depending
              on league size) for a genuine role change, and use small bids to win
              uncontested adds.
            </p>
            <p>
              Rolling priority rewards patience and punishes panic. If your league uses
              it, do not burn the top claim on a player you could have had for a
              mid-priority add. Save the claim for a starter-level role that will not
              last more than a week on waivers.
            </p>
            <h2 class="static-section-title">What is actually worth a claim</h2>
            <ul style="margin-left:20px; line-height:1.8;">
              <li>A running back who just inherited a backfield because of injury,
                  with the snaps to match, not just a highlight carry. Snap share
                  above 50% on a real offense is the signal; a long touchdown run is
                  the noise.</li>
              <li>A receiver or tight end whose target share jumped after a
                  depth-chart change, which you can verify in
                  <a href="/guides/reading-advanced-metrics">advanced metrics</a>.
                  Two consecutive weeks above 20% target share is a role. One week at
                  25% is a data point.</li>
              <li>In Superflex, a quarterback who is suddenly starting. Format
                  scarcity makes streaming passers more valuable than in 1QB. Even a
                  mediocre starter has a floor that no waiver receiver can match.</li>
              <li>In dynasty, a young player who just earned a real NFL role even if
                  the short-term points are modest. That is future
                  <a href="/guides/dynasty-trade-value">trade value</a>. A 22-year-old
                  who just became a full-time starter is a better dynasty add than a
                  29-year-old who just had a two-touchdown game.</li>
            </ul>
            <p>
              What is rarely worth a big bid: a one-week dart throw with no path to
              snaps once the starter returns, a player you will not start or stash,
              and a name from a national recap that every other manager also saw. If
              the whole league is bidding, the price reflects the hype, not the role.
              Let someone else pay for the headline.
            </p>
            <h2 class="static-section-title">How to size a bid</h2>
            <p>
              Most managers bid from gut feel, which is why most FAAB mistakes are
              predictable. A workable framework: assign every target to one of three
              tiers. Tier one is a locked-in multi-week starter, the kind of add that
              changes your lineup. That is worth a real bid, often 25% or more of your
              remaining budget in shallow leagues. Tier two is a useful stash or
              streamer with a plausible path: 5 to 15%. Tier three is a lottery ticket
              or a one-week fill-in: the minimum, or nothing at all.
            </p>
            <p>
              Adjust for timing: a tier-one add in week three has more seasonal value
              than the same profile in week twelve, so early-season bids can run
              hotter.
            </p>
            <p>
              One more discipline: decide your drop before you bid. If you cannot name
              the player you are cutting, you do not actually have room, and you will
              end up cutting the new add two weeks later after paying for him. The
              <a href="/compare">compare tool</a> is useful here: stack the target
              against your worst bench player and make sure the add is actually an
              upgrade.
            </p>
            <h2 class="static-section-title">The Tuesday process: verify before you spend</h2>
            <p>
              The wire rewards managers who move on Tuesday with a plan, not managers
              who bid on every trending name. Build a short weekly routine. First,
              scan the <a href="/top-movers">top movers</a> and your league&rsquo;s
              trending adds to see who the market likes. Second, check the usage
              behind the name: snaps, routes, and target share on the
              <a href="/metrics">advanced metrics</a> page, plus the
              <a href="/breakouts">breakout board</a> for young players whose
              opportunity is trending up. Third, ask the role question: will this
              player&rsquo;s snaps still exist in a month? If the answer depends on a
              starter staying hurt with no timetable, discount the bid.
            </p>
            <h2 class="static-section-title">Dynasty-specific waiver habits</h2>
            <p>
              Taxi squads change the math. A rookie who is not startable this year can
              still be a correct add if he has draft capital and a path. Do not clog
              taxi with 28-year-old lottery tickets. Prefer the same traits you would
              want in a <a href="/guides/rookie-draft-strategy">rookie draft</a>: age,
              capital, and opportunity.
            </p>
            <p>
              In dynasty, every wire add should pass a second test: could this player
              be worth something in a trade by next season? A young player earning
              snaps has a rising
              <a href="/dynasty-trade-value-chart">trade value</a> even before the
              points arrive, which makes him currency. A veteran streamer has no
              future value at all. When the two are close for this week&rsquo;s
              lineup, the young player is usually the better dynasty add, because he
              appreciates while the veteran only depreciates.
            </p>
            <p>
              Also mind your roster&rsquo;s life cycle. A
              <a href="/guides/contending-in-dynasty">contender</a> should spend wire
              capital on production that helps now. A
              <a href="/guides/dynasty-rebuild-strategy">rebuilder</a> should spend it
              on youth with paths to roles, even if the payoff is a year away. The
              worst dynasty wire habit is a rebuilder burning FAAB on a 30-year-old
              streamer and a contender stashing a practice-squad rookie. Match the add
              to the mission.
            </p>
            <h2 class="static-section-title">Walkthrough: a real decision, step by step</h2>
            <p>
              Imagine it is Wednesday morning. A backup running back just rushed for
              110 yards after the starter left with an injury, and he is the
              most-added player in your league. Here is the process. First, check the
              snap share: he played 58% of snaps after the injury, not 25%. Good.
              Second, check the injury news: the starter is expected to miss four to
              six weeks. That is a real window, not a one-week rental. Third, check
              the schedule and the offense: a middling offense with a soft next month
              is fine; a bad offense means the volume may not convert. Fourth, set
              the tier: locked-in starter for a month, so tier one. Fifth, name your
              drop: your WR6, who has not cracked 40% of snaps all season. Sixth, bid
              with conviction but not panic: a strong bid, sized for your league,
              placed early in the week before the hype cycle peaks.
            </p>
            <h2 class="static-section-title">Mistakes that drain a FAAB budget</h2>
            <ul style="margin-left:20px; line-height:1.8;">
              <li><strong>Bidding on the headline, not the role.</strong> If the usage
                  did not change, the production will not repeat. Check snaps and
                  targets first, every time.</li>
              <li><strong>Saving everything for a rainy day.</strong> Unspent FAAB does
                  not score points. If a genuine role change appears in week five and
                  you are sitting on a full budget &ldquo;just in case,&rdquo; that
                  was the case.</li>
              <li><strong>Ignoring your drop.</strong> A great add attached to a
                  terrible drop is a sideways move. Know who you are cutting before
                  you bid.</li>
            </ul>
            <div class="highlight-box">
              Bid on roles that will still exist next month. Spend real FAAB rarely,
              and never spend it on a headline without usage behind it.
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
              Average Draft Position (ADP) is a record of where players have been taken
              in recent drafts. Dynasty trade value is an estimate of what those
              players are worth in trades right now. They are related, and they often
              disagree, and that disagreement is useful if you know which question you
              are answering.
            </p>
            <h2 class="static-section-title">Two numbers, two different questions</h2>
            <p>
              ADP answers one question: &ldquo;when will this player be gone?&rdquo;
              Trade value answers a different one: &ldquo;what is he worth right
              now?&rdquo; The first is a clock for draft day. The second is a price for
              every other day of the year.
            </p>
            <p>
              Mixing them up is expensive. Using trade value to decide when to draft a
              player means reaching two rounds early for a name the room would have
              let fall. Using ADP to judge a trade means pricing a deal with last
              month&rsquo;s news. Each number is excellent at its own job and mediocre
              at the other&rsquo;s.
            </p>
            <h2 class="static-section-title">What ADP is good for</h2>
            <p>
              ADP is a clock. In a startup, it tells you the latest you can reasonably
              wait on a player before someone else takes him. In a rookie draft, it
              tells you which names are likely gone at the turn. That timing function
              is ADP&rsquo;s superpower: nothing else tells you where your specific
              draft room draws its lines.
            </p>
            <p>
              ADP is also a measure of consensus hype, which is useful information
              even when you disagree with it. If the room is drafting a 30-year-old
              receiver two rounds above your board, let someone else pay the hype tax.
              And when you want to know whether your league will pay up for a certain
              quarterback, ADP tells you exactly how the crowd feels. Hype you can see
              is hype you can exploit, usually by fading it.
            </p>
            <h2 class="static-section-title">Where ADP misleads</h2>
            <p>
              ADP is a weak measure of true talent. Draft rooms are public, social,
              and slow to update after injuries and depth-chart news. A player can
              climb ADP for three weeks on a narrative while his snap share does not
              move. That is why BR Fantasy treats market data as one input into
              <a href="/guides/dynasty-trade-value">trade value</a>, not the whole
              model.
            </p>
            <p>
              ADP also lags role changes by design. It is an average of drafts that
              already happened, so it describes where the market <em>was</em>, not
              where it is going. A second-year receiver who earned a full-time role in
              week 3 will still show September ADP in October. The draft already
              happened. The trade market has moved on, and ADP cannot tell you by how
              much.
            </p>
            <h2 class="static-section-title">What trade value is good for</h2>
            <p>
              Trade value is for deals, keep-or-cut decisions, and comparing two
              players who will never share a draft room again. It can move daily with
              production and market trades. If you are asking &ldquo;should I accept
              this package?&rdquo; you want the
              <a href="/dynasty-trade-value-chart">value chart</a> and the
              <a href="/trade">trade calculator</a>, not last month&rsquo;s startup
              ADP.
            </p>
            <p>
              Trade value is a weak start/sit tool. A high dynasty number can belong
              to a young player you should bench this week, because the number prices
              his next four years and your lineup needs this Sunday. Do not start the
              expensive name over the boring volume earner because the chart said so.
            </p>
            <h2 class="static-section-title">When they diverge, look for a reason</h2>
            <ul style="margin-left:20px; line-height:1.8;">
              <li><strong>ADP higher than trade value:</strong> the player is a
                  draft-room celebrity. Fine to let him go in a startup if your board
                  prefers a less-famous role player. Risky to buy him in-season at ADP
                  prices. This is the classic name-brand tax: the room remembers last
                  year&rsquo;s finish and has not priced in the new competition for
                  targets.</li>
              <li><strong>Trade value higher than ADP:</strong> the managers who
                  already own him will not sell cheap, even if drafters have not
                  caught up. That often happens after a mid-season breakout, when the
                  role changed after most drafts were held. See
                  <a href="/top-movers">top movers</a>. Chasing these players at the
                  new value is fine; expecting to buy them at ADP is how offers get
                  ignored.</li>
              <li><strong>Both falling:</strong> believe the role change until metrics
                  say otherwise. Do not &ldquo;buy the name&rdquo; just because ADP
                  used to be high. A falling ADP plus a falling trade value means the
                  market and the drafters agree something changed: usually snaps,
                  targets, or health.</li>
            </ul>
            <h2 class="static-section-title">Rookie ADP is its own animal</h2>
            <p>
              Rookie ADP is especially noisy before the NFL draft and again after
              training camp. Pre-draft boards rank prospects on college production and
              athletic testing. Post-draft, landing spot reshuffles everything: the
              same prospect can be a league-winner or an afterthought depending on
              depth-chart competition. Then camp reports move ADP again, often on
              beat-writer enthusiasm rather than depth-chart reality.
            </p>
            <p>
              Use <a href="/prospects">prospect profiles</a> and landing-spot context,
              not just a big board screenshot from May. And see
              <a href="/guides/rookie-draft-strategy">rookie draft strategy</a> for how
              to value the picks themselves once the noise settles.
            </p>
            <h2 class="static-section-title">Putting both to work</h2>
            <p>
              In a startup, draft with the ADP clock and pick with your board. ADP
              tells you that your target receiver usually lasts until round 6; your
              board tells you he is worth a round 4 pick. You wait until round 5 and
              take the discount. Trade value&rsquo;s job in a startup is quieter: it
              tells you which &ldquo;reaches&rdquo; are real reaches. If your board
              loves a player the value chart prices two rounds above ADP, that is not
              a reach, it is an inefficiency. Take it and let the room wonder.
            </p>
            <p>
              In-season, trade by value and forget ADP exists. By October, ADP is a
              photograph of a room that no longer exists. A mid-season breakout&rsquo;s
              ADP still says &ldquo;round 9 pick&rdquo; while his trade value says
              &ldquo;low-end WR2,&rdquo; and the manager who owns him knows which
              number matters. Check <a href="/top-movers">top movers</a> for where
              value is traveling, confirm the role on the
              <a href="/metrics">advanced metrics page</a>, and make your offer from
              the current price. The
              <a href="/guides/buy-low-sell-high">buy-low, sell-high</a> windows live
              exactly in these gaps between stale perception and current value.
            </p>
            <div class="highlight-box">
              Use ADP to time a draft pick. Use trade value to judge a deal. Mixing
              the two is how you overpay in March and under-sell in October.
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
              Dynasty value is a bet on remaining seasons, not just remaining weeks.
              That is why two players with similar projections this year can sit in
              completely different tiers on a
              <a href="/dynasty-trade-value-chart">trade value chart</a>. Aging is not
              a cliff you notice on a birthday. It is a slow change in injury risk,
              role, and how NFL teams choose to use a player. The curve is different
              at every position.
            </p>
            <p>
              The practical consequence: stop pricing every 27-year-old the same. A
              27-year-old receiver with a locked-in target share is a core asset. A
              27-year-old running back with 1,500 career touches is a depreciating
              one. This guide gives you the shape of each position&rsquo;s curve and,
              more importantly, how to turn it into decisions.
            </p>
            <h2 class="static-section-title">Running backs: the steepest curve</h2>
            <p>
              Running backs take the most wear and have the shortest peak. Many still
              produce at 26 and 27; far fewer are weekly RB1s at 29. The decline rarely
              announces itself with one bad season. It shows up first as lost goal-line
              work, then as a committee role, then as a release. By the time the box
              score looks bad, the market has already moved.
            </p>
            <p>
              That does not mean you should never roster an aging back. It means their
              correct price is a rental: valuable to a
              <a href="/guides/contending-in-dynasty">contender</a>, dangerous as a
              cornerstone in a
              <a href="/guides/dynasty-rebuild-strategy">rebuild</a>. When an older
              back is having a career year, that is often the
              <a href="/guides/buy-low-sell-high">sell-high</a> window, not the time to
              extend your window around him. Check whether the production is coming
              from a huge snap share you cannot count on next year.
            </p>
            <p>
              Workload history matters as much as the birthday. A 26-year-old with
              three seasons of 300-touch work has more tread worn off than a
              26-year-old who spent two years in a committee. Age sets the
              neighborhood; career touches pick the house.
            </p>
            <h2 class="static-section-title">Receivers: a longer prime, with exceptions</h2>
            <p>
              Wide receivers as a group hold production deeper into their late 20s,
              especially high-volume players who do not rely on elite speed alone.
              Route-running and chemistry with the quarterback age better than pure
              speed, which is why possession receivers routinely outlast burners. A
              27-year-old WR1 with sticky target share is still a building block.
            </p>
            <p>
              But a 27-year-old gadget receiver who lives on big plays is not the same
              aging profile just because the birthdays match. Role determines the
              curve within the position: every-down target earners decay slowly, while
              situational deep threats fall off suddenly when one step goes. Price the
              role&rsquo;s aging curve, not the position&rsquo;s average.
            </p>
            <p>
              Young receivers can take a year to earn targets, so be careful in the
              other direction too. Paying up for a 22-year-old with draft capital and a
              clear path is usually better than paying up for a 24-year-old who has
              already failed to earn a role. Metrics help separate those cases; see
              <a href="/guides/reading-advanced-metrics">reading advanced metrics</a>.
            </p>
            <h2 class="static-section-title">Quarterbacks: format decides the price</h2>
            <p>
              Passers age more slowly than skill players, which is why veterans remain
              useful even as their dynasty number cools. A quarterback&rsquo;s value is
              tied to starting seasons, and starters routinely play into their
              mid-30s. In Superflex, a 32-year-old starter can still be a high-value
              asset because the replacement cost is so steep. In 1QB, that same player
              is often a cheap stabilizer. Details in
              <a href="/guides/superflex-vs-1qb">Superflex vs 1QB</a>.
            </p>
            <p>
              Age still matters for the true elites you would build around for five
              years; it matters less for the QB12 you are starting this season. The
              common mistake is treating a 34-year-old QB2 like a depreciating asset
              to dump at any price in Superflex. Scarcity beats age at the position. A
              locked-in starting quarterback&rsquo;s floor has value that no
              24-year-old backup can replace.
            </p>
            <h2 class="static-section-title">Tight ends: late bloomers, then a cliff</h2>
            <p>
              Tight ends often break out later than receivers, so patience in year two
              and three is rational, especially in
              <a href="/guides/te-premium-leagues">TE premium</a>. Once they arrive,
              they can hold value into their late 20s, because the position rewards
              size, technique, and red-zone chemistry more than raw speed.
            </p>
            <p>
              The players to be careful with are 28-plus tight ends whose role is
              blocking-first with occasional scoring spikes. Those spikes are redraft
              candy and dynasty traps: a two-touchdown game does not change a 45%
              route share. Pay for every-down tight ends, not touchdown tourists.
            </p>
            <h2 class="static-section-title">Turning age into decisions: a checklist</h2>
            <p>
              BR Fantasy&rsquo;s values bake positional aging into the daily model so
              you do not have to apply a homemade discount in every trade. You still
              have to decide whether <em>your</em> window matches the player&rsquo;s
              remaining prime. Run every acquisition through these questions:
            </p>
            <ul style="margin-left:20px; line-height:1.8;">
              <li><strong>How many peak seasons are you buying?</strong> A 28-year-old
                  back gives a contender one or two. A 24-year-old WR2 gives a
                  rebuilder five. Price the seasons, not the name.</li>
              <li><strong>Does the role match the age?</strong> An every-down role
                  extends every curve. A situational role shortens it. A 29-year-old
                  every-down receiver is a safer hold than a 26-year-old gadget
                  player.</li>
              <li><strong>What does the workload history say?</strong> Heavy career
                  touches move running backs down their curve faster than birthdays
                  suggest. Check the odometer, not just the birth certificate.</li>
              <li><strong>Who replaces the production when it ends?</strong> If your
                  answer is a rookie pick that does not exist yet, you are rebuilding
                  around hope. Old players are fine as bridges; they are fatal as
                  foundations.</li>
              <li><strong>Is the market already pricing the decline?</strong>
                  Sometimes it is, and the veteran is fairly cheap. Age discounts are
                  only an edge when the market over-applies them.</li>
            </ul>
            <h2 class="static-section-title">When to break your own age rules</h2>
            <p>
              Rules are for the average case. Break them when the price already
              reflects the age: a 29-year-old WR1 with an intact target share selling
              at a 26-year-old&rsquo;s discount is a buy for anyone, because the
              market did your discounting for you. Break them for timeline fit: a
              contender buying a 28-year-old running back for a second-round pick is
              not violating the curve, he is renting exactly what the curve says the
              player still has.
            </p>
            <p>
              Break them in the other direction too. Youth is not a defense for a
              player who never earned a role. A 23-year-old receiver entering year
              three with a 12% target share is not young, he is a veteran of failing
              to get open. The curve describes players with roles. Without a role,
              there is nothing to age gracefully.
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
              hit first: put players (and picks) on two sides, see how the values
              compare, and decide whether to accept, reject, or counter. It is not
              trying to replace your judgment. It is trying to stop you from
              negotiating from a screenshot of last year&rsquo;s rankings.
            </p>
            <h2 class="static-section-title">Step 1: match the calculator to your league</h2>
            <p>
              Before you add names, set the format. Superflex vs 1QB changes
              quarterback prices more than any other toggle. Team count and scoring
              (including TE premium) also move the result. If those settings are
              wrong, the grade is wrong, and you will either insult a league-mate with
              a lowball or accept a deal that only looks fair on a default chart.
              Background on why the numbers differ is in
              <a href="/guides/dynasty-trade-value">how dynasty trade value works</a>
              and <a href="/guides/superflex-vs-1qb">Superflex vs 1QB</a>.
            </p>
            <p>
              If you have already connected a league, the calculator can use your
              roster context. That is the difference between a generic &ldquo;side A
              wins by 8%&rdquo; and a deal that actually fills a hole you have.
              Connecting is optional for a quick public look; it is worth it if you
              trade often.
            </p>
            <h2 class="static-section-title">Step 2: build both sides completely</h2>
            <p>
              The most common calculator error is not the settings, it is an
              incomplete board. Include rookie picks on the side that is actually
              sending them. A &ldquo;player for player&rdquo; grade that forgets the
              extra second-round pick is how people think they won a trade that the
              rest of the league immediately calls a fleece.
            </p>
            <p>
              Add every piece, even the throw-ins. A third-round pick will not swing a
              grade much, but two thirds and a young bench stash add up, and leaving
              pieces off trains you to undervalue your own depth. If a deal has more
              than three players a side, grade the core pieces first, then add the
              depth to see whether it moves the needle.
            </p>
            <h2 class="static-section-title">Step 3: read the grade, then ignore it slightly</h2>
            <p>
              A balanced grade means the market thinks the packages are close, not
              that the trade is right for you. A lopsided grade means you should have
              a reason to continue: consolidation, a win-now running back, or a
              rebuild unpack. Those reasons are spelled out in
              <a href="/guides/evaluating-a-trade">how to evaluate a dynasty trade</a>.
              If you cannot name one, the grade is the answer. The calculator is the
              scale; that guide is the recipe.
            </p>
            <p>
              Treat small gaps as ties. A grade that says you are down 4% is not
              telling you to reject; it is telling you the packages are in the same
              band and the decision belongs to fit. When a single name is doing most
              of the work in a grade, look up that player&rsquo;s page for the
              long-form value context, then come back to the two-sided grade.
            </p>
            <h2 class="static-section-title">Step 4: turn a no into a counter</h2>
            <p>
              Counters exist because first offers are rarely final. Say the calculator
              shows you down about 12% on a deal you otherwise like: you send a
              30-valued receiver for a 26-valued running back plus a 20-valued young
              tight end. Add the smallest piece that closes the gap, a future third or
              a bench receiver with a pulse, instead of a message calling the other
              side crazy.
            </p>
            <p>
              The psychology matters as much as the math. Deals close when both
              managers can defend the result in the group chat. A counter that moves
              the grade from minus 12% to minus 3% while giving the other manager a
              story lands far more often than a technically perfect demand that makes
              them feel fleeced. If the gap is 30% or more, do not counter with spare
              parts: walk away or rebuild the package around a different centerpiece.
            </p>
            <h2 class="static-section-title">Step 5: picks break lazy grades</h2>
            <p>
              Rookie picks are the easiest pieces to misprice because they are options
              on players who do not exist yet. A future first is worth roughly a
              proven young starter in most formats, but its exact value depends on
              where it lands, which nobody knows. Price it as a mid first unless you
              have real reason to think it will be early or late.
            </p>
            <p>
              Never grade a &ldquo;player for player&rdquo; deal as close when a pick
              is attached and uncounted. And when you are the one receiving picks,
              check <a href="/guides/rookie-draft-strategy">rookie draft strategy</a>
              so you know what a first, second, and third actually buy. A first buys
              a real prospect. A third buys a lottery ticket. The calculator knows the
              difference; make sure you do too.
            </p>
            <h2 class="static-section-title">Step 6: when to override the grade</h2>
            <p>
              The calculator prices the market. You price your roster. Override a
              balanced grade when the deal clearly fits your window: a
              <a href="/guides/contending-in-dynasty">contender</a> should happily lose
              5% of raw value consolidating two flex pieces into one every-week
              starter, because starting spots are limited and the best players are
              hardest to replace. A
              <a href="/guides/dynasty-rebuild-strategy">rebuilder</a> should happily
              lose 5% unpacking an aging star into younger pieces.
            </p>
            <p>
              Also override for age curves the market has not fully priced. A
              28-year-old running back and a 24-year-old receiver can grade evenly
              while aging in opposite directions; see
              <a href="/guides/positional-aging-curves">positional aging curves</a>.
              The grade is a snapshot. Your job is to know which direction each asset
              is traveling.
            </p>
            <h2 class="static-section-title">Step 7: common calculator mistakes</h2>
            <ul style="margin-left:20px; line-height:1.8;">
              <li><strong>Wrong format toggle.</strong> Grading a Superflex deal on
                  1QB values under-prices every quarterback involved. Check the toggle
                  every time, not just the first time.</li>
              <li><strong>Chasing exact zero.</strong> A deal does not need to grade
                  at 0.0% to be good. Fit beats symmetry.</li>
              <li><strong>Ignoring direction.</strong> Check whether each name is
                  rising or falling on <a href="/top-movers">top movers</a> and
                  confirm usage with
                  <a href="/guides/reading-advanced-metrics">advanced metrics</a>
                  before you buy a dip that is actually a lost role.</li>
              <li><strong>Sending the cornerstone deal at midnight.</strong> Sleep on
                  any trade that moves a foundational player. Dynasty deals are rarely
                  so urgent that you must accept before the next snap, and if the other
                  manager is rushing you, that is information too.</li>
              <li><strong>Forgetting your league&rsquo;s norms.</strong> The
                  calculator cannot see that your league never trades first-round
                  picks in-season, or that one manager will not sell a certain player
                  at any price. Use the tool for the market half of the decision,
                  then be a human for the rest.</li>
            </ul>
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
