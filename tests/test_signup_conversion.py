"""Signup conversion polish: hero funnel order, social proof, OAuth prominence,
post-signup claim fast path, and honest pricing copy.

Social proof is the verbatim Jayden Waddell testimonial already on the site.
No user counts are shown anywhere: none are available server-side, and we do
not invent numbers.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
DASH_CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
LITE_CSS = (ROOT / "static" / "landing_lite.css").read_text(encoding="utf-8")
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
BILLING_PY = (ROOT / "routes" / "billing_bp.py").read_text(encoding="utf-8")
GOOGLE_AUTH_PY = (ROOT / "routes" / "auth_bp.py").read_text(encoding="utf-8")

QUOTE = "THATS ACTUALLY SO SICK BRO"


def _hero() -> str:
    # A+ landing: the hero section (headline card + matchup board) plus the
    # connect card (.home-hero-right), which sits after the editorial
    # sections in DOM order (AdSense Tier 2). Together they hold the funnel
    # content the old in-hero card held.
    hero_start = APP_PY.index('<section class="home-hero')
    hero_end = APP_PY.index("</section>", hero_start) + len("</section>")
    card_start = APP_PY.index('<div class="home-hero-right">', hero_end)
    card_end = APP_PY.index('<div class="home-content-wrapper">', card_start)
    return APP_PY[hero_start:hero_end] + APP_PY[card_start:card_end]


def _plain(text: str) -> str:
    """Tag-stripped, entity-unescaped text (the quote is styled with a span)."""
    import html as _html
    import re as _re

    return _html.unescape(_re.sub(r"<[^>]+>", "", text))


# ── Hero funnel order: free path (Get started) before the PRO CTA ────────────

def test_get_started_card_comes_before_pro_cta():
    # The connect card follows the hero directly (the approved mock's
    # order): a scrolling visitor reaches Get started before the tools
    # strip, the quote, the PRO pitch, the freeband and the FAQ. It used
    # to sit after the FAQ, ~5,000px down the phone page, and was
    # effectively unfindable without the hero CTA.
    hero = _hero()
    assert "Get started" in hero
    body = APP_PY[APP_PY.index("FORM_BODY"):]
    order = [body.index(marker) for marker in (
        '<div class="home-hero-right">', "Built for the whole week",
        "ld-quote", 'id="homeProSignup"', '<section class="freeband',
        '<section class="faq">')]
    assert order == sorted(order)


def test_right_column_stacks_card_over_cta():
    for css in (DASH_CSS, LITE_CSS):
        block = css[css.index(".home-hero-right {"):css.index(".home-hero-right {") + 220]
        assert "flex-direction: column" in block


# ── Social proof near CTAs: verbatim testimonial, never invented numbers ──────

def test_hero_has_testimonial_near_cta():
    # The quote now lives in its own section between the demos and PRO.
    assert "ld-quote" in APP_PY
    assert QUOTE in _plain(APP_PY)
    assert "Jayden Waddell" in APP_PY
    assert APP_PY.index("ld-quote") < APP_PY.index('id="homeProSignup"')

def test_pricing_page_has_testimonial_near_plan_ctas():
    assert "pricing-proof" in BILLING_PY
    assert QUOTE in BILLING_PY
    assert "Jayden Waddell" in BILLING_PY


def test_social_proof_uses_the_existing_verbatim_quote():
    # The quote must match the site's testimonial character for character:
    # no paraphrase, no invented attribution. It renders once, in the
    # landing's quote section (styled with a span, so compare plain text).
    assert '<section class="quotes">' not in APP_PY
    assert "home-hero-proof" not in APP_PY
    assert _plain(APP_PY).count(QUOTE) == 1
    assert "Jayden Waddell" in APP_PY



def test_no_invented_user_counts_in_new_proof():
    quote = APP_PY[APP_PY.index("ld-quote"):APP_PY.index("</figure>", APP_PY.index("ld-quote"))]
    for snippet in ("managers strong", "happy managers", "users", "members"):
        assert snippet not in quote


def test_no_em_dashes_in_new_copy():
    for snippet in (
        "THATS ACTUALLY SO SICK BRO",
        "From $10 a year.",
        "Only the AI storyline is PRO",
        "Claim your team",
        "Sign in with Google",
        "Connect your first league",
    ):
        assert "\u2014" not in snippet


# ── OAuth prominence on sign-in surfaces ──────────────────────────────────────

def test_guest_card_sign_in_is_a_full_google_button():
    assert 'class="google-continue-btn" href="/auth/google?intent=login' in APP_PY
    assert "Sign in with Google" in APP_PY
    assert "home-signin-link" not in APP_PY


def test_guest_access_still_prominent():
    hero = _hero()
    assert "No account needed to look around" in hero
    assert "Connect your league below" in hero
    # The free trust strip right under the hero is untouched.
    assert "Connect your league below. No account needed to look around." in APP_PY


def test_oauth_state_nonce_pkce_untouched():
    for token in (
        "google_oauth_state",
        "google_oauth_nonce",
        "google_pkce_verifier",
        "code_challenge_method",
        "S256",
    ):
        assert token in GOOGLE_AUTH_PY


# ── Post-signup claim-your-team fast path ─────────────────────────────────────

def test_zero_league_signin_drops_into_claim_flow():
    # Anchor on the claim-flow copy, not the bare `if (!leagues.length) {`
    # (the push-settings backfill added another one earlier in the file).
    anchor = 'signedInLeagueList.textContent = "Connect your first fantasy league below."'
    at = APP_JS.index(anchor)
    branch = APP_JS[at - 200:at + 400]
    assert 'setHomeCardState("connect")' in branch
    assert "Connect your first league" in branch


# ── Honest pricing / claim copy ───────────────────────────────────────────────

def test_pricing_copy_matches_pricing_page():
    home = APP_PY[APP_PY.index('FORM_BODY = """'):]
    assert "Less than $1 a month" not in home
    assert "From $10 a year." in home


def test_weekly_recap_copy_scopes_pro_to_storyline():
    assert "Only the AI storyline is PRO" in APP_PY


def test_claim_team_modal_title():
    assert "Claim your team" in APP_PY
    assert "Sign in to your team" not in APP_PY
