"""
Public / static-content routes (no league context required).

Routes: /privacy, /support, /faq, /contact, /about, /terms, /sw.js, /ads.txt
Also handles the league-context variants: /<platform>/<season>/<league_id>/...
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from flask import Blueprint, send_file, request

public_bp = Blueprint("public", __name__)


def _render(title: str, league_id: Optional[str], active: str, body: str,
            platform: Optional[str] = None, season: Optional[int] = None) -> str:
    # Late import avoids circular dependency at module load time.
    from app import render_page
    return render_page(title, league_id, active, body, platform, season)


# ── Service worker ────────────────────────────────────────────────────────────

@public_bp.route("/sw.js")
def service_worker():
    return send_file("static/sw.js", mimetype="application/javascript")


# ── Ads.txt ───────────────────────────────────────────────────────────────────

@public_bp.route("/ads.txt")
def ads_txt():
    ads_file = Path(__file__).resolve().parents[1] / "ads.txt"
    if ads_file.exists():
        return send_file(ads_file, mimetype="text/plain")
    return "# ads.txt - Add your ad network credentials here", 200, {"Content-Type": "text/plain"}


# ── Robots.txt ────────────────────────────────────────────────────────────────

@public_bp.route("/robots.txt")
def robots_txt():
    base = request.host_url.rstrip("/")
    body = (
        "User-agent: *\n"
        "Allow: /\n"
        "Disallow: /api/\n"
        "Disallow: /set-viewer\n"
        "Disallow: /logout\n"
        "\n"
        f"Sitemap: {base}/sitemap.xml\n"
    )
    return body, 200, {"Content-Type": "text/plain; charset=utf-8"}


# ── Sitemap.xml ───────────────────────────────────────────────────────────────

@public_bp.route("/sitemap.xml")
def sitemap_xml():
    from app import get_nfl_state
    from datetime import datetime
    base = request.host_url.rstrip("/")
    nfl_state = get_nfl_state() or {}
    season = int(nfl_state.get("season") or datetime.now().year)

    # Static pages always indexed
    static_urls = [
        ("", "1.0", "daily"),
        ("/trade", "0.9", "daily"),
        ("/dynasty-trade-value-chart", "0.9", "weekly"),
        ("/top-movers", "0.8", "weekly"),
        ("/rankings/dynasty", "0.8", "weekly"),
        ("/rankings/dynasty-qb", "0.8", "weekly"),
        ("/rankings/dynasty-rb", "0.8", "weekly"),
        ("/rankings/dynasty-wr", "0.8", "weekly"),
        ("/rankings/dynasty-te", "0.8", "weekly"),
        ("/trade-intel", "0.8", "daily"),
        ("/trade-database", "0.8", "weekly"),
        ("/players", "0.7", "weekly"),
        ("/breakouts", "0.7", "weekly"),
        ("/prospects", "0.7", "weekly"),
        ("/pricing", "0.6", "monthly"),
        ("/privacy", "0.3", "monthly"),
        ("/faq", "0.4", "monthly"),
        ("/support", "0.4", "monthly"),
        ("/contact", "0.3", "monthly"),
        ("/about", "0.4", "monthly"),
        ("/terms", "0.3", "monthly"),
        ("/guides", "0.7", "monthly"),
    ]

    # Guide articles (original long-form content)
    for _slug in _GUIDE_ORDER:
        static_urls.append((f"/guides/{_slug}", "0.6", "monthly"))

    import xml.etree.ElementTree as ET
    urlset = ET.Element("urlset", xmlns="http://www.sitemaps.org/schemas/sitemap/0.9")
    for path, priority, changefreq in static_urls:
        url_el = ET.SubElement(urlset, "url")
        ET.SubElement(url_el, "loc").text = base + path
        ET.SubElement(url_el, "priority").text = priority
        ET.SubElement(url_el, "changefreq").text = changefreq

    # Per-player trade-value pages
    try:
        from app import get_player_slug_index
        for slug in sorted(get_player_slug_index().keys()):
            url_el = ET.SubElement(urlset, "url")
            ET.SubElement(url_el, "loc").text = f"{base}/player/{slug}/trade-value"
            ET.SubElement(url_el, "priority").text = "0.7"
            ET.SubElement(url_el, "changefreq").text = "weekly"
    except Exception:
        pass  # never let player-page enumeration break the sitemap

    xml_bytes = ET.tostring(urlset, encoding="unicode", xml_declaration=False)
    body = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_bytes
    return body, 200, {"Content-Type": "application/xml; charset=utf-8"}


# ── Privacy ───────────────────────────────────────────────────────────────────

@public_bp.route("/privacy")
@public_bp.route("/<platform>/<int:season>/<league_id>/privacy")
def privacy_page(platform: Optional[str] = None, season: Optional[int] = None,
                 league_id: Optional[str] = None):
    body = """
        <div class="static-page">
          <div class="static-card-page">

            <h1 class="static-hero-title">Privacy Policy</h1>
            <div class="static-section">
              <div class="static-section-title">What We Collect</div>
              <p>
                We use your Sleeper league ID and public Sleeper data to build dashboards,
                projections, and tools. No passwords, payment info, or sensitive personal data
                is collected.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">What We Don't Collect</div>
              <p>
                We don't store personal identifying information, sell data, or track you outside
                of this site.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">Data Storage</div>
              <p>
                League data is cached temporarily on the server to improve performance.
                You may request removal at any time via the Contact page.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">Trade Analytics</div>
              <p>
                When you enter your Sleeper username, your connected league IDs and Sleeper
                user ID may be used to improve trade value accuracy across the platform.
                This data is not sold or shared with third parties.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">Advertising</div>
              <p>
                This site displays advertisements through Google AdSense. Google uses cookies
                to serve ads based on your prior visits to this site or other websites.
                Google's use of advertising cookies enables it and its partners to serve ads
                based on your visit to this site and/or other sites on the Internet.
              </p>
              <p style="margin-top:8px;">
                You may opt out of personalized advertising by visiting
                <a href="https://www.google.com/settings/ads" target="_blank" rel="noopener">
                  Google's Ads Settings
                </a> or
                <a href="http://www.aboutads.info/choices/" target="_blank" rel="noopener">
                  www.aboutads.info
                </a>.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">Cookies</div>
              <p>
                We use cookies to maintain your login session and improve your experience.
                Third-party vendors, including Google, also use cookies to serve ads based
                on your browsing activity. By using this site, you consent to the use of
                cookies as described in this policy.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">Third-Party Links</div>
              <p>
                Our site may contain links to external websites. We are not responsible
                for the privacy practices or content of these third-party sites.
              </p>
            </div>

            <div class="highlight-box">
              Have questions or want your league data removed?
              Reach out using the Contact page.
            </div>

          </div>
        </div>
        """
    return _render("BR Fantasy Privacy", league_id or None, "privacy", body, platform, season)


# ── Support ───────────────────────────────────────────────────────────────────

@public_bp.route("/support")
@public_bp.route("/<platform>/<int:season>/<league_id>/support")
def support_page(platform: Optional[str] = None, season: Optional[int] = None,
                 league_id: Optional[str] = None):
    body = """
        <div class="static-page">
          <div class="static-card-page">
            <h1 class="static-hero-title">Support the Site</h1>

            <div class="static-section">
              <div class="static-section-title">1. Direct Support</div>
              <p>
                If you find the dashboard helpful for your league, you can support
                ongoing development and hosting costs.
              </p>
              <p style="margin-top:6px;">
                <a
                  class="link-pill"
                  href="https://buymeacoffee.com/brfantasy"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  💸 Make a donation
                </a>
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">2. Go Premium</div>
              <p>
                Premium unlocks AI-powered insights, breakout candidate rankings, advanced
                metrics, and a trade intelligence database &mdash; either for your whole league
                or just your own account across all leagues.
              </p>
              <p style="margin-top:8px;">
                Visit the <strong>Settings</strong> menu inside your league dashboard to
                subscribe, or reach out on the Contact page with any questions.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">3. Share With Your League</div>
              <p>
                Honestly one of the best ways to support this is just using it.
                Share the link with your league mates, show the dashboards on stream,
                or use the matchup previews in your weekly recaps.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">4. Follow & Subscribe</div>
              <div style="display:flex; gap:10px; flex-wrap:wrap;">
                <a class="link-pill" href="https://youtube.com/@hoodiekj" target="_blank">▶️ YouTube</a>
                <a class="link-pill" href="https://twitch.tv/hoodiekj1" target="_blank">🎮 Twitch</a>
                <a class="link-pill" href="https://twitter.com/hoodiekj16" target="_blank">🐦 Twitter/X</a>
              </div>
            </div>

            <div class="highlight-box">
              Every bit of support helps keep the site online and evolving for future seasons.
              Thanks for using BR Fantasy.
            </div>
          </div>
        </div>
        """
    return _render("BR Fantasy Support", league_id or None, "support", body, platform, season)


# ── FAQ ───────────────────────────────────────────────────────────────────────

@public_bp.route("/faq")
@public_bp.route("/<platform>/<int:season>/<league_id>/faq")
def faq_page(platform: Optional[str] = None, season: Optional[int] = None,
             league_id: Optional[str] = None):
    body = """
        <div class="static-page">
          <div class="static-card-page">
            <h1 class="static-hero-title">FAQ</h1>

            <div class="static-section">
              <div class="static-section-title">General</div>

              <details class="faq-item" open>
                <summary>What is the BR Fantasy Dashboard?</summary>
                <p>
                  It's a custom fantasy football dashboard that pulls in your Sleeper league
                  data and turns it into power rankings, weekly summaries, matchup previews,
                  graphs, and more-all in one place.
                </p>
              </details>

              <details class="faq-item">
                <summary>What do I need to use it?</summary>
                <p>
                  All you need is your Sleeper or ESPN league ID. Paste it into the home screen,
                  and the dashboard will fetch public data for that league.
                </p>
              </details>

              <details class="faq-item">
                <summary>Does this change anything in my Fantasy league?</summary>
                <p>
                  No. The dashboard is read-only. It just reads public data from your league's
                  API and never modifies your league, rosters, or settings.
                </p>
              </details>
            </div>

            <div class="static-section">
              <div class="static-section-title">Rankings & Values</div>

              <details class="faq-item">
                <summary>How are dynasty trade values calculated?</summary>
                <p>
                  Values come from a hybrid model that blends consensus market data with recent
                  production, age curves, and opportunity metrics, recalculated daily. For a full
                  breakdown, read <a href="/guides/dynasty-trade-value">How Dynasty Trade Value
                  Works</a>, or browse the live <a href="/rankings/dynasty">dynasty rankings</a>.
                </p>
              </details>

              <details class="faq-item">
                <summary>Do values change for Superflex leagues?</summary>
                <p>
                  Yes &mdash; significantly. Quarterbacks are far more valuable in Superflex. The
                  rankings and <a href="/trade">trade calculator</a> adjust to your format, and
                  <a href="/guides/superflex-vs-1qb">this guide</a> explains why the same player can
                  have two very different values.
                </p>
              </details>

              <details class="faq-item">
                <summary>How often are rankings updated?</summary>
                <p>
                  Player values and rankings are recalculated daily so they reflect the latest
                  usage, injuries, and market movement. The <a href="/top-movers">top movers</a>
                  page highlights the biggest risers and fallers.
                </p>
              </details>
            </div>

            <div class="static-section">
              <div class="static-section-title">Data & Privacy</div>

              <details class="faq-item">
                <summary>What data do you store?</summary>
                <p>
                  Some league data may be cached temporarily so pages load quickly
                  (rosters, users, scores, projections, etc.). We do not store your
                  password or payment information. See the Privacy Policy for more details.
                </p>
              </details>

              <details class="faq-item">
                <summary>Can I have my league data removed?</summary>
                <p>
                  Yes. Use the Contact page to send your Sleeper league ID and request
                  removal. We'll clear cached data for that league.
                </p>
              </details>
            </div>

            <div class="static-section">
              <div class="static-section-title">Premium / Ads / Support</div>

              <details class="faq-item">
                <summary>Is there a premium or ad-free mode?</summary>
                <p>
                  Yes - league and personal subscriptions are available. See the Pricing page
                  for details on what's included.
                </p>
              </details>

              <details class="faq-item">
                <summary>How can I support the site?</summary>
                <p>
                  You can support the project through a premium subscription, donations,
                  or by sharing the site with your league mates.
                  Visit the Support page for options.
                </p>
              </details>
            </div>

            <div class="static-section">
              <div class="static-section-title">Issues & Feedback</div>

              <details class="faq-item">
                <summary>The numbers look wrong-what should I do?</summary>
                <p>
                  First, hit the refresh button on the nav to clear cached data for your
                  league. If something still looks off, send a message via the Contact
                  page with your league ID and a short description of the issue.
                </p>
              </details>

              <details class="faq-item">
                <summary>Can I request new features?</summary>
                <p>
                  Absolutely. This project is built for fantasy degenerates.
                  Drop your ideas on the Contact page and they might make it onto the roadmap.
                </p>
              </details>
            </div>
          </div>
        </div>
        """
    return _render("BR Fantasy FAQ", league_id or None, "faq", body, platform, season)


# ── Contact ───────────────────────────────────────────────────────────────────

@public_bp.route("/contact")
@public_bp.route("/<platform>/<int:season>/<league_id>/contact")
def contact_page(platform: Optional[str] = None, season: Optional[int] = None,
                 league_id: Optional[str] = None):
    body = """
        <div class="static-page">
          <div class="static-card-page">

            <h1 class="static-hero-title">Contact</h1>

            <div class="static-section">
              <div class="static-section-title">Reach us</div>
              <p>The fastest way to get a response is through Discord. You can also follow along on YouTube and X for site updates and dynasty content.</p>

              <div style="display:flex; flex-wrap:wrap; gap:10px; margin-top:12px;">
                <a class="contact-social-pill" href="https://discord.gg/7aZrs7qfur" target="_blank" rel="noopener">
                  <img src="/static/images/discord-brands-solid.png" style="width:16px;height:16px;" alt="">
                  Join Discord
                </a>
                <a class="contact-social-pill" href="https://youtube.com/@hoodiekj" target="_blank" rel="noopener">
                  <img src="/static/images/youtube-brands-solid.png" style="width:16px;height:16px;" alt="">
                  YouTube
                </a>
                <a class="contact-social-pill" href="https://x.com/hoodiekj" target="_blank" rel="noopener">
                  <img src="/static/images/x-twitter-brands-solid.png" style="width:16px;height:16px;" alt="">
                  X / Twitter
                </a>
              </div>
            </div>

            <div class="static-section">
              <div class="static-section-title">What to include</div>
              <ul style="margin-left:20px; font-size:14px;">
                <li>Your Sleeper league ID</li>
                <li>Which page you were on</li>
                <li>What wasn't working or looked incorrect</li>
                <li>Screenshots if possible</li>
              </ul>
            </div>

            <div class="highlight-box">
              Feedback helps shape future features - thanks for helping improve BR Fantasy.
            </div>

          </div>
        </div>
        """
    return _render("BR Fantasy Contact", league_id or None, "contact", body, platform, season)


# ── About ─────────────────────────────────────────────────────────────────────

@public_bp.route("/about")
@public_bp.route("/<platform>/<int:season>/<league_id>/about")
def about_page(platform: Optional[str] = None, season: Optional[int] = None,
               league_id: Optional[str] = None):
    body = """
        <div class="static-page">
          <div class="static-card-page">

            <h1 class="static-hero-title">About BR Fantasy</h1>

            <div class="static-section">
              <div class="static-section-title">What Is This?</div>
              <p>
                BR Fantasy is a custom fantasy football dashboard built on top of public
                Sleeper API data. It transforms raw league stats into power rankings,
                weekly matchup previews, trade analysis, graphs, standings, and more &mdash;
                all in one place.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">Who Built It?</div>
              <p>
                This project was created by
                <a href="https://youtube.com/@hoodiekj" target="_blank" rel="noopener">hoodiekj</a>,
                a fantasy football player and developer who wanted better tools than what
                the major platforms provide. What started as a personal league tool has
                grown into a full-featured dashboard used by leagues across the country.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">Features</div>
              <ul style="margin-left:20px; font-size:14px; line-height:1.8;">
                <li>Power rankings &amp; weekly performance scoring</li>
                <li>Matchup previews with projected scores</li>
                <li>Roster &amp; trade intelligence</li>
                <li>Dynasty breakout candidate rankings</li>
                <li>Full standings, graphs, and historical data</li>
                <li>Trade database powered by real league activity</li>
              </ul>
            </div>

            <div class="static-section">
              <div class="static-section-title">How Our Values Are Calculated</div>
              <p>
                Dynasty and redraft player values are produced by a hybrid model that blends
                consensus market data with recent on-field production, age and positional aging
                curves, and opportunity metrics like target share and snap counts. Values are
                recalculated daily and calibrated against the broader dynasty market, with
                guardrails that prevent any single noisy input from over- or under-rating a player.
                The result is a calibrated estimate of what a player is actually worth in
                league-to-league trades &mdash; not just who scores the most points.
              </p>
              <p style="margin-top:8px;">
                Want to understand the numbers? Our <a href="/guides">strategy guides</a> explain
                <a href="/guides/dynasty-trade-value">how dynasty trade value works</a>,
                <a href="/guides/superflex-vs-1qb">why Superflex changes everything</a>, and
                <a href="/guides/reading-advanced-metrics">how to read advanced metrics</a>.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">Supported Platforms</div>
              <p>
                BR Fantasy works with <strong>Sleeper</strong> and <strong>ESPN</strong> leagues
                today, with Yahoo support on the roadmap. It supports dynasty, keeper, and redraft
                formats, in both single-quarterback and Superflex configurations &mdash; values and
                tools adjust automatically to your league's settings.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">Free vs Premium</div>
              <p>
                The core dashboards, rankings, and trade tools are free. Premium unlocks
                AI-powered trade analysis, the breakout candidate engine, deeper advanced metrics,
                and a trade intelligence database built from real league activity. See the
                <a href="/pricing">Pricing</a> page for details.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">Data Sources</div>
              <p>
                All fantasy data is sourced from public Sleeper and ESPN APIs.
                Player projections and rankings are sourced from publicly available
                fantasy football data providers. No private or proprietary data
                is used without permission.
              </p>
            </div>

            <div class="highlight-box">
              Questions, feedback, or feature ideas? Reach out on the
              <a href="/contact">Contact</a> page.
            </div>

          </div>
        </div>
        """
    return _render("About BR Fantasy", league_id or None, "about", body, platform, season)


# ── Terms ─────────────────────────────────────────────────────────────────────

@public_bp.route("/terms")
@public_bp.route("/<platform>/<int:season>/<league_id>/terms")
def terms_page(platform: Optional[str] = None, season: Optional[int] = None,
               league_id: Optional[str] = None):
    body = """
        <div class="static-page">
          <div class="static-card-page">

            <h1 class="static-hero-title">Terms of Service</h1>

            <div class="static-section">
              <div class="static-section-title">Acceptance of Terms</div>
              <p>
                By accessing or using BR Fantasy, you agree to be bound by these Terms
                of Service. If you do not agree to these terms, please do not use the site.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">Use of the Service</div>
              <p>
                BR Fantasy is provided for personal, non-commercial use. You agree not to
                misuse the service, attempt to gain unauthorized access to any part of the
                system, or use automated tools to scrape or overload the site.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">Data &amp; Privacy</div>
              <p>
                By using this site you acknowledge that league data entered (such as
                Sleeper league IDs and usernames) may be cached and used to improve
                platform-wide features like trade analytics. See the
                <a href="/privacy">Privacy Policy</a> for full details.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">Intellectual Property</div>
              <p>
                All original code, design, and content on BR Fantasy are the property of
                their respective creators. Sleeper&reg; is a registered trademark of
                Sleeper Inc. and is not affiliated with BR Fantasy.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">Disclaimer of Warranties</div>
              <p>
                BR Fantasy is provided &ldquo;as is&rdquo; without warranties of any kind.
                Projections, rankings, and trade values are for informational purposes only
                and do not constitute financial or sports betting advice. We make no
                guarantees about the accuracy, reliability, or availability of the service.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">Limitation of Liability</div>
              <p>
                BR Fantasy and its operators shall not be liable for any indirect,
                incidental, or consequential damages arising from your use of the site.
                Your use of BR Fantasy is at your own risk.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">Changes to These Terms</div>
              <p>
                We reserve the right to update these Terms at any time. Continued use of
                the site after changes are posted constitutes acceptance of the new Terms.
              </p>
            </div>

            <div class="highlight-box">
              Questions about these terms? Reach out on the
              <a href="/contact">Contact</a> page.
            </div>

          </div>
        </div>
        """
    return _render("BR Fantasy Terms", league_id or None, "terms", body, platform, season)


# ── Guides ──────────────────────────────────────────────────────────────────────
# Original, long-form strategy content. This is the public, crawlable "value"
# layer that lives outside the league-context login wall (the dashboards, trade
# calculator, etc. require a league ID, so search engines never see them). Each
# guide is unique editorial copy and links into the public tools/rankings pages
# so crawlers can discover the rest of the site from here.

GUIDES = {
    "dynasty-trade-value": {
        "title": "How Dynasty Trade Value Works",
        "summary": "What a dynasty trade value actually measures, why it differs from "
                   "redraft rankings, and how to read the numbers behind a deal.",
        "body": """
            <p>
              Every player in a dynasty league carries a <strong>trade value</strong> &mdash; a
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
              <li><strong>Consensus market data</strong> &mdash; where the crowd of dynasty managers
                  is actually pricing a player.</li>
              <li><strong>Recent on-field production</strong> &mdash; usage, efficiency, and role,
                  which move a player's stock week to week.</li>
              <li><strong>Age and position curve</strong> &mdash; running backs decline early, wide
                  receivers and quarterbacks hold value far longer.</li>
              <li><strong>Situation</strong> &mdash; target share, depth-chart competition, and team
                  context that affect future opportunity.</li>
            </ul>
            <p>
              You can see calibrated values for every relevant player on the
              <a href="/rankings/dynasty">dynasty rankings</a> page, or browse the full
              <a href="/dynasty-trade-value-chart">dynasty trade value chart</a> to compare across
              positions at a glance.
            </p>
            <div class="static-section-title">Why two values for the same player?</div>
            <p>
              Most dynasty leagues are either single-quarterback (1QB) or Superflex, and a player's
              value can change dramatically between the two formats. Quarterbacks are far more
              valuable in Superflex because you can start two of them. If your league is Superflex,
              always look at Superflex values &mdash; using 1QB numbers will badly under-rate every
              passer. We cover this in depth in
              <a href="/guides/superflex-vs-1qb">Superflex vs 1QB</a>.
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
        "body": """
            <p>
              The single biggest factor in a player's dynasty value &mdash; bigger than age,
              bigger than last week's stat line &mdash; is often just your league format. In a
              <strong>single-quarterback (1QB)</strong> league you start one QB. In a
              <strong>Superflex</strong> league you can start a second quarterback in a flex spot,
              which makes the position enormously more valuable.
            </p>
            <div class="static-section-title">Why quarterbacks explode in Superflex</div>
            <p>
              There are only 32 starting NFL quarterbacks, and in a 12-team Superflex league up to
              24 of them can be in starting lineups every week. That scarcity means even mid-tier
              starters carry real weight, and the elite young passers become the most valuable
              assets in the entire player pool &mdash; frequently worth more than any running back
              or receiver.
            </p>
            <p>
              In 1QB, the opposite is true: you only need one quarterback, streamable options are
              everywhere, and so the position is heavily discounted. Top-tier wide receivers and
              running backs sit at the top of 1QB value charts instead.
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
            <div class="highlight-box">
              Rule of thumb: in Superflex, treat startable quarterbacks as premium assets. In 1QB,
              let the other manager overpay for them.
            </div>
        """,
    },
    "reading-advanced-metrics": {
        "title": "Reading Advanced Metrics: A Fantasy Manager's Guide",
        "summary": "Target share, air yards, snap counts, red-zone usage and more &mdash; what "
                   "each metric tells you and which ones actually predict fantasy points.",
        "body": """
            <p>
              Box-score stats tell you what already happened. <strong>Advanced metrics</strong>
              tell you whether it is likely to keep happening. They separate players who are
              producing because of genuine, repeatable opportunity from those riding unsustainable
              efficiency or touchdown luck. Here's how to read the ones that matter.
            </p>
            <div class="static-section-title">Opportunity metrics (the most predictive)</div>
            <ul style="margin-left:20px; line-height:1.8;">
              <li><strong>Target share</strong> &mdash; the percentage of his team's targets a
                  receiver or tight end earns. A rising target share is one of the strongest
                  leading indicators of future fantasy production.</li>
              <li><strong>Snap share</strong> &mdash; how often a player is actually on the field.
                  Low snap share caps a player's ceiling no matter how efficient he looks.</li>
              <li><strong>Air yards</strong> &mdash; the total downfield distance of a player's
                  targets. High air yards signal a player is being used in a high-value role even
                  before the catches show up.</li>
              <li><strong>Red-zone usage</strong> &mdash; touches and targets inside the 20.
                  Red-zone volume drives touchdowns, which are the most volatile (and valuable)
                  source of fantasy points.</li>
            </ul>
            <div class="static-section-title">Efficiency metrics (context, not gospel)</div>
            <p>
              Yards per route run, yards after catch, and yards per touch describe how well a
              player converts opportunity into production. They are useful, but efficiency is far
              noisier than volume &mdash; a great yards-per-touch number on five touches a game
              won't survive a larger sample. Always weigh efficiency against the opportunity behind
              it.
            </p>
            <div class="static-section-title">How to use them together</div>
            <p>
              The players worth buying are the ones whose opportunity is climbing before the
              fantasy points catch up: rising snaps, rising target share, growing red-zone role.
              That gap between opportunity and output is exactly what the
              <a href="/breakouts">breakout engine</a> is built to surface, and you can dig into
              the underlying numbers on the <a href="/players">player database</a>.
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
        "body": """
            <p>
              The rookie draft is where dynasty championships are quietly built. Cheap, ascending
              young talent is the best value in the format &mdash; but rookie picks are also where
              managers most often overpay for hype. Here's a framework for drafting well.
            </p>
            <div class="static-section-title">Value the picks, then the players</div>
            <p>
              Before you fall in love with a prospect, understand what the pick itself is worth.
              Early first-round rookie picks carry significant trade value because of their upside,
              but that value drops quickly as you move into the second and third rounds. Knowing the
              market price of a pick keeps you from trading a proven player for a lottery ticket.
            </p>
            <div class="static-section-title">What actually predicts rookie success</div>
            <ul style="margin-left:20px; line-height:1.8;">
              <li><strong>Draft capital</strong> &mdash; where the NFL drafted a player is one of
                  the best predictors of opportunity. Teams invest snaps and targets in the players
                  they spent premium picks on.</li>
              <li><strong>Landing spot</strong> &mdash; the same prospect can be a league-winner or
                  a redraft afterthought depending on depth-chart competition and offensive
                  quality.</li>
              <li><strong>College production at a young age</strong> &mdash; players who dominated
                  early in their college careers (a strong &ldquo;breakout age&rdquo;) hit at higher
                  rates.</li>
              <li><strong>Athletic profile</strong> &mdash; testing scores like RAS provide a floor
                  check, especially at receiver and running back.</li>
            </ul>
            <div class="static-section-title">Position priorities</div>
            <p>
              In most formats, prioritize wide receivers early &mdash; they have the longest dynasty
              shelf life and the highest hit rate near the top of rookie drafts. Running backs offer
              immediate production but age out fast, so they are better targeted by contending teams.
              In Superflex, a rookie quarterback with a clear path to starting can be worth a top
              pick on its own.
            </p>
            <p>
              You can study full prospect profiles &mdash; college metrics, draft capital, athletic
              scores, and live ADP movement &mdash; on the <a href="/prospects">rookie prospects</a>
              page.
            </p>
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
        "body": """
            <p>
              Dynasty trade value is not static &mdash; it moves constantly with injuries, depth
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
            <div class="static-section-title">When to sell high</div>
            <ul style="margin-left:20px; line-height:1.8;">
              <li>A player riding an unsustainable touchdown rate that his opportunity won't
                  support.</li>
              <li>An aging running back coming off a big stretch &mdash; sell the name before the
                  cliff.</li>
              <li>A backup who spiked in value during a short injury fill-in for a starter who is
                  about to return.</li>
            </ul>
            <div class="static-section-title">Let the data find the windows</div>
            <p>
              The clearest buy-low and sell-high signals show up as movement in value over time.
              The <a href="/top-movers">top movers</a> page tracks which players are rising and
              falling fastest, and <a href="/trade-intel">trade intelligence</a> surfaces market
              signals from real league activity. Pair those with the
              <a href="/rankings/dynasty">current rankings</a> to spot gaps between a player's price
              and his true outlook.
            </p>
            <div class="highlight-box">
              The market overreacts to recent results. Your edge is patience: buy the dip on talent,
              sell the spike on age and luck.
            </div>
        """,
    },
    "evaluating-a-trade": {
        "title": "How to Evaluate a Dynasty Trade",
        "summary": "A step-by-step process for judging any trade offer &mdash; beyond just adding "
                   "up the values on each side.",
        "body": """
            <p>
              Adding up trade values on each side of a deal is a useful first check, but it is only
              the beginning. The best trades aren't always the ones that &ldquo;win&rdquo; on raw
              value &mdash; they're the ones that make <em>your</em> roster better for <em>your</em>
              timeline. Here's a repeatable process.
            </p>
            <div class="static-section-title">Step 1: Check the raw value</div>
            <p>
              Start by comparing the total value on each side using format-appropriate numbers
              (1QB or Superflex). A quick way to do this is the
              <a href="/trade">trade calculator</a>, which grades both sides and suggests counters.
              If a deal is wildly lopsided on value, you usually have your answer.
            </p>
            <div class="static-section-title">Step 2: Account for consolidation</div>
            <p>
              Two good players are generally worth more than three mediocre ones, because starting
              lineup spots are limited and the best players are the hardest to replace. When you
              trade multiple pieces for one stud, expect &mdash; and accept &mdash; paying a small
              value premium for that consolidation.
            </p>
            <div class="static-section-title">Step 3: Match the deal to your timeline</div>
            <p>
              Are you contending or rebuilding? Contenders should trade youth and picks for proven,
              win-now production. Rebuilders should do the reverse: sell aging stars for young
              players and draft capital. A trade that's &ldquo;fair&rdquo; on value can still be
              wrong if it doesn't fit where your team is in its cycle.
            </p>
            <div class="static-section-title">Step 4: Value positional scarcity and need</div>
            <p>
              A player is worth more to a roster that needs his position. Don't trade from a
              position of strength into another position of strength &mdash; address real lineup
              holes. In Superflex, weigh quarterback depth especially heavily (see
              <a href="/guides/superflex-vs-1qb">Superflex vs 1QB</a>).
            </p>
            <div class="static-section-title">Step 5: Look past this week</div>
            <p>
              Before you finalize, sanity-check the underlying trends from the
              <a href="/guides/reading-advanced-metrics">advanced metrics</a> and the
              <a href="/top-movers">top movers</a> page. You want to be buying ascending players and
              selling declining ones, not the reverse.
            </p>
            <div class="highlight-box">
              A good trade makes your starting lineup better for your timeline. Value is the
              starting point; fit is the decision.
            </div>
        """,
    },
}

_GUIDE_ORDER = [
    "dynasty-trade-value",
    "superflex-vs-1qb",
    "reading-advanced-metrics",
    "rookie-draft-strategy",
    "buy-low-sell-high",
    "evaluating-a-trade",
]


def _guides_base(platform, season, league_id):
    """Path prefix so guide links keep league context when present."""
    if league_id and platform and season:
        return f"/{platform}/{season}/{league_id}"
    return ""


@public_bp.route("/guides")
@public_bp.route("/<platform>/<int:season>/<league_id>/guides")
def guides_index(platform: Optional[str] = None, season: Optional[int] = None,
                 league_id: Optional[str] = None):
    base = _guides_base(platform, season, league_id)
    cards = []
    for slug in _GUIDE_ORDER:
        g = GUIDES[slug]
        cards.append(f"""
            <a class="guide-card" href="{base}/guides/{slug}"
               style="display:block;text-decoration:none;border:1px solid var(--border);
                      border-radius:12px;padding:16px 18px;margin-bottom:12px;">
              <div class="static-section-title" style="margin:0 0 4px;">{g['title']}</div>
              <p style="margin:0;color:var(--muted);font-size:14px;">{g['summary']}</p>
            </a>
        """)
    body = f"""
        <div class="static-page">
          <div class="static-card-page">
            <h1 class="static-hero-title">Dynasty &amp; Fantasy Football Guides</h1>
            <div class="static-section">
              <p>
                Free, in-depth guides to dynasty strategy &mdash; how trade values work, how to read
                advanced metrics, rookie-draft strategy, and how to win trades. Pair them with our
                live <a href="{base}/rankings/dynasty">dynasty rankings</a> and
                <a href="{base}/trade">trade calculator</a> to put the ideas into practice.
              </p>
            </div>
            <div class="static-section">
              {''.join(cards)}
            </div>
          </div>
        </div>
    """
    return _render("Dynasty Fantasy Football Guides", league_id or None, "guides", body, platform, season)


@public_bp.route("/guides/<slug>")
@public_bp.route("/<platform>/<int:season>/<league_id>/guides/<slug>")
def guide_page(slug: str, platform: Optional[str] = None, season: Optional[int] = None,
               league_id: Optional[str] = None):
    g = GUIDES.get(slug)
    base = _guides_base(platform, season, league_id)
    if not g:
        # Unknown slug → send them to the index rather than 404 into a dead end.
        from flask import redirect
        return redirect(f"{base}/guides")

    # Previous/next links keep crawlers (and readers) moving through the section.
    idx = _GUIDE_ORDER.index(slug) if slug in _GUIDE_ORDER else -1
    nav_links = []
    if idx > 0:
        prev_slug = _GUIDE_ORDER[idx - 1]
        nav_links.append(f'<a href="{base}/guides/{prev_slug}">&larr; {GUIDES[prev_slug]["title"]}</a>')
    if 0 <= idx < len(_GUIDE_ORDER) - 1:
        next_slug = _GUIDE_ORDER[idx + 1]
        nav_links.append(f'<a href="{base}/guides/{next_slug}" style="margin-left:auto;">{GUIDES[next_slug]["title"]} &rarr;</a>')
    nav_html = ""
    if nav_links:
        nav_html = (
            '<div class="static-section" style="display:flex;gap:12px;flex-wrap:wrap;'
            'border-top:1px solid var(--border);padding-top:14px;">'
            + "".join(nav_links) + "</div>"
        )

    body = f"""
        <div class="static-page">
          <div class="static-card-page">
            <p style="margin:0 0 6px;font-size:13px;">
              <a href="{base}/guides">&larr; All guides</a>
            </p>
            <h1 class="static-hero-title">{g['title']}</h1>
            <div class="static-section">
              {g['body']}
            </div>
            {nav_html}
          </div>
        </div>
    """
    return _render(f"{g['title']} | BR Fantasy", league_id or None, "guides", body, platform, season)
