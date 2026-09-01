"""
Public / static-content routes (no league context required).

Routes: /privacy, /support, /faq, /contact, /about, /terms, /sw.js, /ads.txt
Also handles the league-context variants: /<platform>/<season>/<league_id>/...
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from flask import Blueprint, Response, send_file, request

from routes.guides_content import (
    GUIDE_AUTHOR_NAME,
    GUIDE_AUTHOR_URL,
    GUIDE_ORDER as _GUIDE_ORDER,
    GUIDES,
)

public_bp = Blueprint("public", __name__)

# Public support inbox (also used as the VAPID contact identity elsewhere).
# Override with CONTACT_EMAIL in the environment if the mailbox changes.
_CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "admin@brfantasy.com").strip()


def _seo_origin() -> str:
    """Canonical https origin for robots/sitemap URLs.

    Prefers PRIMARY_DOMAIN (same normalization as app.py) so the sitemap always
    advertises canonical URLs even when crawled via the onrender.com host; falls
    back to the request origin (https-correct behind the proxy via ProxyFix).
    """
    pd = os.environ.get("PRIMARY_DOMAIN", "").strip().lower()
    if pd.startswith("www."):
        pd = pd[4:]
    if pd:
        return f"https://{pd}"
    return request.host_url.rstrip("/")


def _render(title: str, league_id: Optional[str], active: str, body: str,
            platform: Optional[str] = None, season: Optional[int] = None,
            description: str = "", canonical: Optional[str] = None) -> str:
    # Late import avoids circular dependency at module load time.
    from app import render_page
    return render_page(title, league_id, active, body, platform, season,
                       description=description, canonical=canonical)


# ── Weekly-email unsubscribe (signed, no login) ───────────────────────────────

@public_bp.route("/email/unsubscribe", methods=["GET", "POST"])
def email_unsubscribe():
    """One-click unsubscribe from weekly digest emails. The token is an HMAC over
    the account id, so no login is needed and the link can't be forged.

    GET renders a confirmation page (the link in the email body). POST is the
    RFC 8058 one-click path a mail client hits from the List-Unsubscribe header;
    it just performs the opt-out and returns a bare 200."""
    from flask import request as _req
    from utils.weekly_email import verify_unsub_token, unsubscribe
    token = _req.args.get("token", "") or (_req.form.get("token", "") if _req.method == "POST" else "")
    account_id = verify_unsub_token(token)

    if _req.method == "POST":
        if account_id is not None:
            unsubscribe(account_id)
        return ("", 200)

    if account_id is None:
        msg = ("<h2 style='margin:0 0 8px;'>Link expired or invalid</h2>"
               "<p style='color:var(--text-muted);font-size:14px;'>We couldn't verify "
               "this unsubscribe link. You can manage email preferences from your "
               "account settings.</p>")
    elif unsubscribe(account_id):
        msg = ("<h2 style='margin:0 0 8px;'>You're unsubscribed</h2>"
               "<p style='color:var(--text-muted);font-size:14px;'>You won't receive "
               "any more weekly digest emails. You can re-enable them anytime from "
               "your account settings.</p>")
    else:
        msg = ("<h2 style='margin:0 0 8px;'>Something went wrong</h2>"
               "<p style='color:var(--text-muted);font-size:14px;'>We couldn't update "
               "your preferences right now. Please try again later.</p>")
    body = (
        "<div class='card central' style='max-width:520px;margin:40px auto;'>"
        f"<div class='card-body' style='padding:28px 24px;text-align:center;'>{msg}</div></div>"
    )
    return _render("Email preferences", None, "", body)


@public_bp.route("/webhooks/brevo/email", methods=["POST"])
def brevo_email_webhook():
    """Brevo transactional events (delivered/opened/clicked/bounce/spam).

    Auth is a shared secret (query ``secret``, ``X-Brevo-Webhook-Secret``, or
    ``Authorization: Bearer``). Configure ``BREVO_WEBHOOK_SECRET`` and the same
    token on the Brevo webhook. The weekly send does not depend on this endpoint.
    """
    import hmac as _hmac
    from flask import jsonify, request as _req
    secret = (os.environ.get("BREVO_WEBHOOK_SECRET") or "").strip()
    if not secret:
        return jsonify({
            "ok": False,
            "error": "webhook_unconfigured",
            "detail": "Set BREVO_WEBHOOK_SECRET and configure the same token on the Brevo webhook.",
        }), 503
    provided = (
        (_req.args.get("secret") or "")
        or (_req.headers.get("X-Brevo-Webhook-Secret") or "")
    )
    auth = _req.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        provided = provided or auth.split(None, 1)[-1].strip()
    if not provided or not _hmac.compare_digest(provided, secret):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    payload = _req.get_json(silent=True)
    events = payload if isinstance(payload, list) else [payload] if isinstance(payload, dict) else []
    from utils.email_events import apply_webhook_payload
    results = [apply_webhook_payload(ev) for ev in events if ev]
    return jsonify({"ok": True, "processed": len(results)})


# ── Service worker ────────────────────────────────────────────────────────────

@public_bp.route("/sw.js")
def service_worker():
    resp = send_file("static/sw.js", mimetype="application/javascript")
    # SW must revalidate promptly so deploys aren't stuck behind CDN/browser cache.
    resp.headers["Cache-Control"] = "no-cache"
    return resp


# ── Ads.txt ───────────────────────────────────────────────────────────────────

@public_bp.route("/ads.txt")
def ads_txt():
    """Authorized digital sellers file for AdSense.

    Served as a tiny in-memory Response (not send_file) with a long public
    Cache-Control so Cloudflare/CDN can keep serving it even when the origin
    is slow — ads.txt timeouts are a common AdSense crawl failure.
    """
    ads_file = Path(__file__).resolve().parents[1] / "ads.txt"
    if ads_file.exists():
        body = ads_file.read_text(encoding="utf-8")
    else:
        body = "# ads.txt - Add your ad network credentials here\n"
    resp = Response(body, mimetype="text/plain; charset=utf-8")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


# ── Robots.txt ────────────────────────────────────────────────────────────────

@public_bp.route("/robots.txt")
def robots_txt():
    base = _seo_origin()
    # Explicitly allow AdSense / AdsBot crawlers so Cloudflare-managed blocks
    # or over-broad Disallow rules never starve ad inventory evaluation.
    body = (
        "User-agent: Mediapartners-Google\n"
        "Allow: /\n"
        "\n"
        "User-agent: AdsBot-Google\n"
        "Allow: /\n"
        "\n"
        "User-agent: *\n"
        "Allow: /\n"
        "Disallow: /api/\n"
        "Disallow: /set-viewer\n"
        "Disallow: /logout\n"
        "\n"
        f"Sitemap: {base}/sitemap.xml\n"
    )
    resp = Response(body, mimetype="text/plain; charset=utf-8")
    resp.headers["Cache-Control"] = "public, max-age=3600"
    return resp


# ── Sitemap.xml ───────────────────────────────────────────────────────────────

@public_bp.route("/sitemap.xml")
def sitemap_xml():
    from app import get_nfl_state
    from datetime import datetime
    base = _seo_origin()
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
        ("/compare", "0.7", "weekly"),
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
        ("/glossary", "0.6", "monthly"),
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

    # Per-player trade-value pages: only the top slice by value. Submitting
    # every slug floods the sitemap with thousands of thin pages that Google
    # reports as "Discovered - currently not indexed" and wastes crawl budget.
    try:
        from app import get_top_player_slugs
        for slug in get_top_player_slugs(300):
            url_el = ET.SubElement(urlset, "url")
            ET.SubElement(url_el, "loc").text = f"{base}/player/{slug}/trade-value"
            ET.SubElement(url_el, "priority").text = "0.7"
            ET.SubElement(url_el, "changefreq").text = "weekly"
    except Exception:
        logging.getLogger(__name__).debug("suppressed exception", exc_info=True)

    xml_bytes = ET.tostring(urlset, encoding="unicode", xml_declaration=False)
    body = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_bytes
    return body, 200, {
        "Content-Type": "application/xml; charset=utf-8",
        "Cache-Control": "public, max-age=3600",
    }


# ── Privacy ───────────────────────────────────────────────────────────────────

@public_bp.route("/privacy")
@public_bp.route("/<platform>/<int:season>/<league_id>/privacy")
def privacy_page(platform: Optional[str] = None, season: Optional[int] = None,
                 league_id: Optional[str] = None):
    body = """
        <div class="static-page">
          <div class="static-card-page">

            <h1 class="static-hero-title">Privacy Policy</h1>
            <p class="static-page-meta">Last updated: August 25, 2026</p>
            <div class="static-section">
              <div class="static-section-title">What We Collect</div>
              <p>
                We process the league IDs, usernames, team names, rosters, scores, and other
                league information you ask us to load from Sleeper, ESPN, or Yahoo. If you create
                an account, we also process your email address, account identifier, linked leagues,
                and subscription status. We do not receive your platform password or full payment
                card number; authentication and payment details are handled by the relevant login
                provider and payment processor.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">How We Use Information</div>
              <p>
                We use this information to provide and secure the service, remember your settings,
                operate subscriptions, answer support requests, measure site performance, and
                improve aggregate fantasy-football analytics. We do not sell your personal
                information.
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
                This site displays advertisements through Google AdSense. Google and its partners
                may use cookies, web beacons, IP addresses, or similar technologies to deliver,
                limit, personalize, and measure ads. Google's use of advertising cookies may enable
                it and its partners to serve ads based on your visits to this site and other sites.
                Ad units are labeled &ldquo;Advertisement&rdquo; so they are distinguishable from
                editorial content and navigation.
              </p>
              <p style="margin-top:8px;">
                You can manage personalized advertising in
                <a href="https://myadcenter.google.com/" target="_blank" rel="noopener noreferrer">Google's My Ad Center</a>,
                opt out of personalized ads at
                <a href="https://adssettings.google.com/" target="_blank" rel="noopener noreferrer">adssettings.google.com</a>,
                and learn how Google uses data on partner sites in
                <a href="https://policies.google.com/technologies/partner-sites" target="_blank" rel="noopener noreferrer">Google's partner-sites policy</a>.
                Visitors in the EEA, UK, or Switzerland are shown a Google-certified consent
                message (Funding Choices) before non-essential advertising cookies are used.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">Cookies</div>
              <p>
                Essential cookies maintain your login session, subscription access, preferences,
                and security. Where required, our consent banner lets you accept, reject, or manage
                non-essential advertising cookies before they are used. You can also change your
                choice through the privacy controls provided on the site or clear cookies in your
                browser. Disabling essential cookies may prevent account features from working.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">Service Providers &amp; Retention</div>
              <p>
                We share information only as needed with providers that host the site, authenticate
                accounts, process payments, deliver notifications, monitor errors, and provide
                advertising or analytics. They process information under their own terms and privacy
                policies. We retain account and transaction records while needed to provide the
                service and meet legal obligations; cached league data and logs are retained only as
                long as reasonably needed for performance, security, and analytics.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">Children's Privacy</div>
              <p>
                BR Fantasy is intended for adults who manage fantasy football leagues. It is not
                directed at children, and we do not knowingly collect personal information from
                anyone under 18. If you believe a minor has provided information to us,
                contact us and we will delete it.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">Your Choices</div>
              <p>
                You may ask us to access, correct, or delete information associated with your
                account or league. You may unlink a league, cancel a subscription, change cookie
                choices, or opt out of personalized advertising. Requests can be made through the
                <a href="/contact">Contact page</a>; we may need to verify that you control the
                relevant account or league.
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
    return _render(
        "BR Fantasy Privacy", league_id or None, "privacy", body, platform, season,
        description="How BR Fantasy collects, uses, and protects your account and league data.",
    )


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
                  Make a donation
                </a>
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">2. Go Premium</div>
              <p>
                Premium unlocks AI-powered insights, breakout candidate rankings, advanced
                metrics, and a trade intelligence database, either for your whole league
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
    return _render(
        "BR Fantasy Support", league_id or None, "support", body, platform, season,
        description="Get help with BR Fantasy: account access, league linking, billing, and feature questions.",
    )


# ── FAQ ───────────────────────────────────────────────────────────────────────

# FAQ content lives in one data structure so the on-page accordion and the
# FAQPage structured data are always generated from the same source (no drift).
# Answers may contain inline links; the JSON-LD uses a plain-text version.
FAQ_SECTIONS = [
    ("General", [
        ("What is the BR Fantasy Dashboard?",
         "It's a custom fantasy football dashboard that pulls in your Sleeper league "
         "data and turns it into power rankings, weekly summaries, matchup previews, "
         "graphs, and more - all in one place."),
        ("What do I need to use it?",
         "All you need is your Sleeper or ESPN league ID. Paste it into the home screen, "
         "and the dashboard will fetch public data for that league."),
        ("Does this change anything in my fantasy league?",
         "No. The dashboard is read-only. It just reads public data from your league's "
         "API and never modifies your league, rosters, or settings."),
    ]),
    ("Rankings &amp; Values", [
        ("How are dynasty trade values calculated?",
         "Values come from a hybrid model that blends consensus market data with recent "
         "production, age curves, and opportunity metrics, recalculated daily. For a full "
         'breakdown, read <a href="/guides/dynasty-trade-value">How Dynasty Trade Value '
         'Works</a>, or browse the live <a href="/rankings/dynasty">dynasty rankings</a>.'),
        ("Do values change for Superflex leagues?",
         "Yes, significantly. Quarterbacks are far more valuable in Superflex. The "
         'rankings and <a href="/trade">trade calculator</a> adjust to your format, and '
         '<a href="/guides/superflex-vs-1qb">this guide</a> explains why the same player can '
         "have two very different values."),
        ("How often are rankings updated?",
         "Player values and rankings are recalculated daily so they reflect the latest "
         'usage, injuries, and market movement. The <a href="/top-movers">top movers</a> '
         "page highlights the biggest risers and fallers."),
        ("What do the fantasy terms on the site mean?",
         'Every abbreviation and strategy term - ADP, Superflex, TE Premium, Zero-RB, '
         'and more - is defined in plain English in the <a href="/glossary">fantasy '
         "football glossary</a>."),
    ]),
    ("Data &amp; Privacy", [
        ("What data do you store?",
         "Some league data may be cached temporarily so pages load quickly "
         "(rosters, users, scores, projections, etc.). We do not store your "
         "password or payment information. See the Privacy Policy for more details."),
        ("Can I have my league data removed?",
         "Yes. Use the Contact page to send your Sleeper league ID and request "
         "removal. We'll clear cached data for that league."),
    ]),
    ("Premium / Ads / Support", [
        ("Is there a premium or ad-free mode?",
         "Yes - league and personal subscriptions are available. See the Pricing page "
         "for details on what's included."),
        ("How can I support the site?",
         "You can support the project through a premium subscription, donations, "
         "or by sharing the site with your league mates. "
         "Visit the Support page for options."),
    ]),
    ("Issues &amp; Feedback", [
        ("The numbers look wrong - what should I do?",
         "First, hit the refresh button on the nav to clear cached data for your "
         "league. If something still looks off, send a message via the Contact "
         "page with your league ID and a short description of the issue."),
        ("Can I request new features?",
         "Absolutely. This project is built for fantasy degenerates. "
         "Drop your ideas on the Contact page and they might make it onto the roadmap."),
    ]),
]


def _faq_plain(html_answer: str) -> str:
    """Strip inline tags/entities for the JSON-LD acceptedAnswer text."""
    import re
    txt = re.sub(r"<[^>]+>", "", html_answer)
    txt = (txt.replace("&mdash;", "-").replace("&amp;", "&")
              .replace("&lt;", "<").replace("&gt;", ">"))
    return re.sub(r"\s+", " ", txt).strip()


@public_bp.route("/faq")
@public_bp.route("/<platform>/<int:season>/<league_id>/faq")
def faq_page(platform: Optional[str] = None, season: Optional[int] = None,
             league_id: Optional[str] = None):
    import json

    sections_html = []
    first = True
    for section, items in FAQ_SECTIONS:
        rows = []
        for q, a in items:
            open_attr = " open" if first else ""
            first = False
            rows.append(
                f'<details class="faq-item"{open_attr}>'
                f'<summary>{q}</summary><p>{a}</p></details>'
            )
        sections_html.append(
            '<div class="static-section">'
            f'<div class="static-section-title">{section}</div>'
            + "".join(rows) + "</div>"
        )

    # FAQPage structured data, built from the same source as the accordion so it
    # can never drift from what's on the page. Escape '<' so an answer's link can
    # never break out of the <script> block.
    faq_ld = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {
                "@type": "Question",
                "name": _faq_plain(q),
                "acceptedAnswer": {"@type": "Answer", "text": _faq_plain(a)},
            }
            for _section, items in FAQ_SECTIONS for q, a in items
        ],
    }
    jsonld = (
        '<script type="application/ld+json">'
        + json.dumps(faq_ld, separators=(",", ":")).replace("<", "\\u003c")
        + "</script>"
    )

    body = (
        '<div class="static-page"><div class="static-card-page">'
        '<h1 class="static-hero-title">FAQ</h1>'
        + "".join(sections_html)
        + jsonld
        + "</div></div>"
    )
    return _render(
        "BR Fantasy FAQ", league_id or None, "faq", body, platform, season,
        description="Answers to common questions about BR Fantasy Football: how "
                    "dynasty trade values are calculated, Superflex scoring, data "
                    "privacy, and more.",
    )


# ── Contact ───────────────────────────────────────────────────────────────────

@public_bp.route("/contact")
@public_bp.route("/<platform>/<int:season>/<league_id>/contact")
def contact_page(platform: Optional[str] = None, season: Optional[int] = None,
                 league_id: Optional[str] = None):
    email = _CONTACT_EMAIL
    body = f"""
        <div class="static-page">
          <div class="static-card-page">

            <h1 class="static-hero-title">Contact</h1>

            <div class="static-section">
              <div class="static-section-title">Reach us</div>
              <p>
                Email is the best way to reach us for privacy requests, corrections, and account
                issues. Discord is fastest for product questions and feedback. You can also follow
                along on YouTube and X for site updates and dynasty content.
              </p>

              <div style="display:flex; flex-wrap:wrap; gap:10px; margin-top:12px;">
                <a class="contact-social-pill" href="mailto:{email}">
                  Email {email}
                </a>
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
    return _render(
        "BR Fantasy Contact", league_id or None, "contact", body, platform, season,
        description="Contact the BR Fantasy team with feedback, bug reports, or partnership questions.",
    )


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
                weekly matchup previews, trade analysis, graphs, standings, and more, 
                all in one place.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">How It Started</div>
              <p>
                As commissioner of a new dynasty league with my friends, I wanted to make it more
                fun. So I started making spreadsheets with weekly stats, showing who scored
                the most, which teams were actually good, and who was trash.
              </p>
              <p style="margin-top:8px;">
                My league mates started interacting way more. We'd talk about someone getting
                carried by Puka because he scored 40, not just that they won. The trash talk got
                better. People wanted to see the data. It made the whole thing feel less like
                &ldquo;check your lineup&rdquo; and more like an actual community.
              </p>
              <p style="margin-top:8px;">
                I was making weekly recap videos, and by Week 8 I was 1&ndash;7, getting abused in
                the chat for being so bad. Every week I'd argue I was getting hosed. Then I ran it
                back 6&ndash;0 and made the playoffs. These videos and the website made the whole
                season better than anything I'd played before.
              </p>
              <p style="margin-top:8px;">
                I liked doing it so much that I just kept expanding it, more graphs, value
                tracking, recaps. Basically built out everything I was doing manually into an
                actual website.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">Why It's Public</div>
              <p>
                I figured other commissioners might want this too, so I made it accessible. If
                you're running a league and want to show people what's actually happening beyond
                just scores, try it, it's free. If you have feedback or ideas, let me know
                on the <a href="/contact">Contact</a> page.
              </p>
              <p style="margin-top:8px;">
                I'm
                <a href="https://youtube.com/@hoodiekj" target="_blank" rel="noopener">hoodiekj</a>,
                a fantasy football player and developer who wanted better tools than the major
                platforms provide. I write the
                <a href="/guides">strategy guides</a> and build and maintain BR Fantasy myself.
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
                league-to-league trades, not just who scores the most points.
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
                BR Fantasy works with <strong>Sleeper</strong>, <strong>ESPN</strong>, and
                <strong>Yahoo</strong> leagues today, plus MFL and Fleaflicker. It supports dynasty, keeper, and redraft
                formats, in both single-quarterback and Superflex configurations, values and
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

            <div class="static-section">
              <div class="static-section-title">Editorial Standards &amp; Corrections</div>
              <p>
                Guides, explanations, rankings, and model commentary are created for BR Fantasy,
                not copied or automatically republished from other sites. Analysis distinguishes
                measured league data from projections and opinion, and methodology is explained
                wherever a model-generated value is presented. Automated outputs are reviewed and
                updated as source data changes; no ranking or projection is a guarantee.
              </p>
              <p style="margin-top:8px;">
                If you spot an inaccurate statistic, broken source, or unclear recommendation,
                report the page and issue through <a href="/contact">Contact</a>. Material errors
                are corrected as soon as they can be verified.
              </p>
            </div>

            <div class="highlight-box">
              Questions, feedback, or feature ideas? Reach out on the
              <a href="/contact">Contact</a> page.
            </div>

          </div>
        </div>
        """
    return _render(
        "About BR Fantasy", league_id or None, "about", body, platform, season,
        description="About BR Fantasy: dynasty trade values, advanced metrics, and multi-platform league tools.",
    )


# ── Terms ─────────────────────────────────────────────────────────────────────

@public_bp.route("/terms")
@public_bp.route("/<platform>/<int:season>/<league_id>/terms")
def terms_page(platform: Optional[str] = None, season: Optional[int] = None,
               league_id: Optional[str] = None):
    body = """
        <div class="static-page">
          <div class="static-card-page">

            <h1 class="static-hero-title">Terms of Service</h1>
            <p class="static-page-meta">Last updated: August 25, 2026</p>

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
              <div class="static-section-title">Advertising</div>
              <p>
                Free pages may display third-party advertisements labeled
                &ldquo;Advertisement.&rdquo; Ads are provided by Google AdSense and its partners
                under their own terms. Premium subscriptions remove ads for covered accounts.
                Do not click ads except to visit an advertiser you are genuinely interested in;
                we never ask users to click ads or offer incentives for doing so.
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
    return _render(
        "BR Fantasy Terms", league_id or None, "terms", body, platform, season,
        description="Terms of use for the BR Fantasy football analytics dashboard and tools.",
    )


# ── Guides ──────────────────────────────────────────────────────────────────────
# Original, long-form strategy content lives in routes/guides_content.py so this
# module stays routing, not a 14-article anthology. Each guide is unique
# editorial copy that search engines and AdSense reviewers can crawl without a
# league login.

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
                Original strategy writing from BR Fantasy: how dynasty trade values are built,
                why Superflex prices quarterbacks so differently from 1QB, how to read
                target share and snap counts, and how to rebuild, contend, and draft without
                copying a generic cheat sheet. These are full articles, not tool captions.
              </p>
              <p style="margin-top:8px;">
                Pair them with live <a href="{base}/rankings/dynasty">dynasty rankings</a>, the
                <a href="{base}/dynasty-trade-value-chart">trade value chart</a>, and the
                <a href="{base}/trade">trade calculator</a>. Keep the
                <a href="{base}/glossary">glossary</a> handy for any unfamiliar terms. No
                account is required to read the guides or use the public rankings.
              </p>
            </div>
            <div class="static-section">
              {''.join(cards)}
            </div>
          </div>
        </div>
    """
    return _render(
        "Dynasty Fantasy Football Guides", league_id or None, "guides", body, platform, season,
        description="Free original dynasty fantasy football guides covering trade value, "
                    "Superflex, advanced metrics, rebuilds, startups, and the waiver wire.",
    )


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

    published = g.get("published") or ""
    updated = g.get("updated") or published
    byline = (
        f'<p class="guide-byline">By <a href="{GUIDE_AUTHOR_URL}" target="_blank" '
        f'rel="author noopener">{GUIDE_AUTHOR_NAME}</a>'
        f'{f" · Updated {updated}" if updated else ""}</p>'
    )

    import json
    origin = _seo_origin()
    article_ld = {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": g["title"],
        "description": g.get("summary") or "",
        "datePublished": published,
        "dateModified": updated,
        "author": {
            "@type": "Person",
            "name": GUIDE_AUTHOR_NAME,
            "url": GUIDE_AUTHOR_URL,
        },
        "publisher": {
            "@type": "Organization",
            "name": "BR Fantasy",
            "logo": {"@type": "ImageObject", "url": f"{origin}/static/BR_Logo.png"},
        },
        "mainEntityOfPage": f"{origin}{base}/guides/{slug}",
    }
    jsonld = (
        '<script type="application/ld+json">'
        + json.dumps(article_ld, separators=(",", ":")).replace("<", "\\u003c")
        + "</script>"
    )

    body = f"""
        <div class="static-page">
          <div class="static-card-page">
            <p style="margin:0 0 6px;font-size:13px;">
              <a href="{base}/guides">&larr; All guides</a>
            </p>
            <h1 class="static-hero-title">{g['title']}</h1>
            {byline}
            <div class="static-section">
              {g['body']}
            </div>
            {nav_html}
            {jsonld}
          </div>
        </div>
    """
    return _render(
        f"{g['title']} | BR Fantasy", league_id or None, "guides", body,
        platform, season, description=g.get("summary") or "",
    )


# ── Glossary ─────────────────────────────────────────────────────────────────
# Plain-English definitions for the dynasty/redraft vocabulary used across the
# site. Each term is unique crawlable content and a long-tail search magnet
# ("what is superflex", "te premium meaning"), and links back into the tools.

GLOSSARY = [
    ("Average Draft Position", "ADP",
     "The average slot a player is taken across many drafts. It's a quick read on "
     "consensus value and tells you roughly when you'll have to reach to get a player."),
    ("Superflex", "SF",
     'A lineup with a flex spot that can start a quarterback, so most teams start two '
     'QBs. It makes passers dramatically more valuable than in 1QB leagues - see '
     '<a href="/guides/superflex-vs-1qb">Superflex vs 1QB</a>.'),
    ("TE Premium", "TEP",
     "Scoring that gives tight ends extra points per reception (often +0.5 PPR). It "
     "lifts every TE's value and widens the gap between the elite tier and everyone else."),
    ("Points Per Reception", "PPR",
     "A scoring format that awards points for each catch (1.0 in full PPR, 0.5 in half). "
     "It boosts high-volume receivers and pass-catching running backs."),
    ("Value Over Replacement", "VOR",
     "How much better a player is than a freely available replacement at his position. "
     "It's the idea behind why scarce positions are drafted earlier than raw points suggest."),
    ("Taxi Squad", None,
     "A reserve area for rookies and young players that keeps them off your active roster "
     "without exposing them to waivers - a dynasty tool for stashing developmental talent."),
    ("Zero-RB", None,
     "A draft strategy that intentionally fades running back early to load up on receivers, "
     "then attacks RB value later in the draft and on the waiver wire."),
    ("Hero-RB", None,
     "A middle-ground strategy: take one elite running back early, then pivot to receivers "
     "and fill the rest of the backfield with upside later."),
    ("Handcuff", None,
     "The backup who would inherit a starter's workload if he got hurt. Handcuffing your "
     "own bell-cow RB insures his weekly volume against injury."),
    ("Bell-Cow", None,
     "A running back who dominates his team's touches - carries, goal-line work, and "
     "passing-down snaps. The most valuable and rarest RB archetype."),
    ("Startup Draft", None,
     "The initial draft that stocks a brand-new dynasty league from the entire player pool, "
     "veterans and rookies together, before any rookie-only drafts happen."),
    ("Rookie Draft", None,
     'A dynasty-only draft held each offseason for that year\'s incoming rookies. See '
     '<a href="/guides/rookie-draft-strategy">rookie draft strategy</a>.'),
    ("Target Share", None,
     "The percentage of a team's passing targets that go to one player. It's one of the "
     "stickiest signals of real receiving opportunity and future production."),
    ("Snap Share", None,
     "The percentage of a team's offensive snaps a player is on the field for. High snap "
     "share signals a secure role even when the box score is quiet."),
    ("Air Yards", None,
     "The total downfield distance of the passes thrown a receiver's way, whether caught or "
     "not. It measures how a player is used and hints at scoring upside."),
    ("Red Zone", "RZ",
     "The area inside an opponent's 20-yard line, where touchdowns are scored. Red-zone "
     "targets and carries are premium opportunities that drive fantasy points."),
    ("Consistency", None,
     "How reliably a player hits his projection week to week. A consistent floor is worth "
     "more to a contender than a boom-or-bust profile with the same average."),
    ("Buy Low / Sell High", None,
     'Trading for a player when his value dips below his true worth, or dealing one away '
     'while the market overrates him. See <a href="/guides/buy-low-sell-high">buy low, '
     "sell high</a>."),
    ("Contending vs Rebuilding", None,
     "The two dynasty modes: a contender trades youth and picks for win-now production; a "
     "rebuilder does the opposite, banking young players and draft capital for later."),
    ("Draft Capital", None,
     "The rookie picks a team controls, treated as a tradeable asset. Future first-round "
     "picks are dynasty currency for both rebuilds and win-now upgrades."),
    ("Dynasty", None,
     "A league format where you keep your entire roster year over year, drafting only "
     "rookies each offseason. It rewards long-term roster building over single-season luck."),
    ("Redraft", None,
     "The classic format where every team drafts a brand-new roster from scratch each season. "
     "Nothing carries over, so only this year's production matters."),
    ("Keeper", None,
     "A hybrid format where you retain a small number of players between seasons, usually at "
     "a draft-pick cost, then redraft the rest."),
    ("Waiver Wire", None,
     "The pool of unrostered players and the rules for claiming them. Waiver priority or a "
     "FAAB budget decides who wins contested adds."),
    ("Free Agent Acquisition Budget", "FAAB",
     "A season-long budget you bid against to claim waiver players, instead of a simple "
     "priority order. Spending it well is a real in-season edge."),
    ("Strength of Schedule", "SOS",
     "How hard a player's or team's upcoming matchups are. It matters most for streaming "
     "decisions and for playoff-week planning."),
    ("Regression", None,
     "The tendency for unsustainable stats (like an extreme touchdown rate) to drift back "
     "toward the norm. Spotting it early is how you sell high before the market catches on."),
    ("Ceiling / Floor", None,
     "A player's best realistic week (ceiling) and worst (floor). Tournaments reward ceiling; "
     "cash and must-win weeks reward a high floor."),
    ("Breakout", None,
     'A young player poised to leap to a new tier of production. Our '
     '<a href="/breakouts">breakouts</a> model flags candidates before the market does.'),
    ("Sleeper", None,
     "A player being drafted well below his upside - low cost, high potential payoff. "
     "(Not to be confused with the Sleeper league platform.)"),
]


@public_bp.route("/glossary")
@public_bp.route("/<platform>/<int:season>/<league_id>/glossary")
def glossary_page(platform: Optional[str] = None, season: Optional[int] = None,
                  league_id: Optional[str] = None):
    import json

    terms = sorted(GLOSSARY, key=lambda t: t[0].lower())

    cards = []
    for term, abbr, definition in terms:
        abbr_html = f'<span class="gloss-abbr">{abbr}</span>' if abbr else ""
        cards.append(
            '<div class="gloss-term">'
            f'<h2 class="gloss-name">{term}{abbr_html}</h2>'
            f'<p class="gloss-def">{definition}</p>'
            "</div>"
        )

    # DefinedTermSet structured data — each entry is a DefinedTerm. Plain-text
    # descriptions; escape '<' so a definition's link can't break the <script>.
    def _plain(s: str) -> str:
        import re
        s = re.sub(r"<[^>]+>", "", s).replace("&mdash;", "-").replace("&amp;", "&")
        return re.sub(r"\s+", " ", s).strip()

    term_set = {
        "@context": "https://schema.org",
        "@type": "DefinedTermSet",
        "name": "Fantasy Football Glossary",
        "hasDefinedTerm": [
            {"@type": "DefinedTerm", "name": (f"{t} ({a})" if a else t),
             "description": _plain(d)}
            for t, a, d in terms
        ],
    }
    jsonld = (
        '<script type="application/ld+json">'
        + json.dumps(term_set, separators=(",", ":")).replace("<", "\\u003c")
        + "</script>"
    )

    body = (
        '<div class="static-page"><div class="static-card-page">'
        '<h1 class="static-hero-title">Fantasy Football Glossary</h1>'
        '<div class="static-section">'
        '<p>Plain-English definitions for the dynasty and redraft terms you\'ll see '
        'across the site. New to a concept? Pair these with our free '
        '<a href="/guides">strategy guides</a> and live '
        '<a href="/rankings/dynasty">dynasty rankings</a>.</p>'
        "</div>"
        f'<div class="gloss-grid">{"".join(cards)}</div>'
        + jsonld
        + "</div></div>"
    )
    return _render(
        "Fantasy Football Glossary | BR Fantasy", league_id or None, "glossary",
        body, platform, season,
        description="Plain-English definitions of dynasty and redraft fantasy "
                    "football terms - ADP, Superflex, TE Premium, Zero-RB, FAAB, "
                    "target share, and more.",
    )
