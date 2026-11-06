"""Pricing consistency fixes (site-improvement check 2026-11-06, finds 1-3).

- The guest-home PRO block must not claim $10/year covers every league:
  $10/year is Starter, which covers exactly 1 league.
- The home PRO block carries a real server-rendered CTA link to /pricing.
- /pricing plan cards are headed by their tier names (Starter / All-Pro /
  Hall of Fame, the names the FAQ and feature copy use) with the league
  count kept as the subline under each name.
- /pricing has a descriptive title and a pricing-specific meta description
  instead of the bare "Pricing" title and the generic site-wide description.
"""
import re
from pathlib import Path

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

import routes.billing_bp as billing

ROOT = Path(__file__).resolve().parents[1]
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")


def _home_pro_block() -> str:
    start = APP_PY.index('id="homeProSignup"')
    return APP_PY[start:APP_PY.index("</section>", start)]


def _plan_cards(html: str):
    start = html.index("pricing-plan-grid")
    end = html.index("pricing-proof")
    return re.findall(
        r'<article class="pricing-option[^"]*" data-plan-card="([^"]+)">(.*?)</article>',
        html[start:end],
        re.S,
    )


# ── Find 1: home PRO price claim ────────────────────────────────────────────

def test_home_pro_block_prices_starter_honestly():
    block = _home_pro_block()
    assert "all PRO tools for 1 league" in block
    assert "all PRO tools, every league" not in block
    assert "every league" not in block


def test_guest_home_renders_honest_pro_line(offline_client):
    html = offline_client.get("/").get_data(as_text=True)
    assert "all PRO tools, every league" not in html
    assert "all PRO tools for 1 league" in html


# ── Find 3a: home PRO block links to /pricing ───────────────────────────────

def test_home_pro_block_links_to_pricing():
    block = _home_pro_block()
    assert '<a class="ld-pro-pricing-link" href="/pricing">' in block
    assert "See all plans and pricing" in block


def test_guest_home_renders_pricing_link(offline_client):
    html = offline_client.get("/").get_data(as_text=True)
    assert 'href="/pricing"' in html


# ── Find 2: pricing cards carry tier names, league count as subline ────────

def test_pricing_cards_headed_by_tier_names(offline_client):
    html = offline_client.get("/pricing").get_data(as_text=True)
    expected = {
        "starter": ("Starter", "1 league"),
        "all_pro": ("All-Pro", "5 leagues"),
        "hall_of_fame": ("Hall of Fame", "Unlimited leagues"),
    }
    cards = _plan_cards(html)
    assert [key for key, _ in cards] == ["starter", "all_pro", "hall_of_fame"]
    for key, body in cards:
        name, leagues = expected[key]
        assert f"<h3>{name}</h3>" in body, key
        assert f'<p class="pricing-leagues">{leagues}</p>' in body, key
        # The tier name heads the card: it appears before the league subline.
        assert body.index(f"<h3>{name}</h3>") < body.index(
            f'<p class="pricing-leagues">{leagues}</p>'), key


# ── Find 3b: pricing title + meta description ───────────────────────────────

def test_pricing_title_and_meta_are_specific(offline_client):
    html = offline_client.get("/pricing").get_data(as_text=True)
    assert "<title>Pricing - BR Fantasy</title>" in html
    assert "<title>Pricing</title>" not in html
    match = re.search(r'<meta name="description" content="([^"]+)"', html)
    assert match, "pricing page is missing a meta description"
    desc = match.group(1)
    assert "Starter is $10/year for 1 league" in desc
    assert "Hall of Fame is $50/year" in desc
    # Not the generic site-wide description.
    assert "free fantasy football toolkit" not in desc


def test_pricing_title_meta_constants_have_no_em_dash():
    assert billing._PRICING_TITLE == "Pricing - BR Fantasy"
    assert "\u2014" not in billing._PRICING_DESCRIPTION
    assert "Starter is $10/year for 1 league" in billing._PRICING_DESCRIPTION
