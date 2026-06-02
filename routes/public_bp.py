"""
Public / static-content routes (no league context required).

Routes: /privacy, /support, /faq, /contact, /sw.js, /ads.txt
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
    base = request.host_url.rstrip("/")
    nfl_state = get_nfl_state() or {}
    season = int(nfl_state.get("season") or 2025)

    # Static pages always indexed
    static_urls = [
        ("", "1.0", "daily"),
        ("/trade", "0.9", "daily"),
        ("/trade-intel", "0.9", "daily"),
        ("/trade-database", "0.8", "weekly"),
        ("/players", "0.8", "weekly"),
        ("/breakouts", "0.8", "weekly"),
        ("/prospects", "0.7", "weekly"),
        ("/pricing", "0.6", "monthly"),
        ("/privacy", "0.3", "monthly"),
        ("/faq", "0.4", "monthly"),
        ("/support", "0.4", "monthly"),
        ("/contact", "0.3", "monthly"),
    ]

    import xml.etree.ElementTree as ET
    urlset = ET.Element("urlset", xmlns="http://www.sitemaps.org/schemas/sitemap/0.9")
    for path, priority, changefreq in static_urls:
        url_el = ET.SubElement(urlset, "url")
        ET.SubElement(url_el, "loc").text = base + path
        ET.SubElement(url_el, "priority").text = priority
        ET.SubElement(url_el, "changefreq").text = changefreq

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

@public_bp.route("/contact", methods=["GET", "POST"])
@public_bp.route("/<platform>/<int:season>/<league_id>/contact", methods=["GET", "POST"])
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
