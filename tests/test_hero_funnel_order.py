"""Guest home hero: free path before PRO CTA; pricing copy matches pricing page;
Weekly Recap PRO scoping; team-claim modal title."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
DASH_CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


def _hero() -> str:
    start = APP_PY.index('<section class="home-hero">')
    end = APP_PY.index("</section>", start)
    return APP_PY[start:end]


def test_get_started_card_comes_before_pro_cta():
    hero = _hero()
    assert "Get started" in hero
    assert "home-pro-hero-cta" in hero
    # The PRO CTA now lives in the right column, under the Get started card.
    assert hero.index("Get started") < hero.index("home-pro-hero-cta")
    left = hero[hero.index("home-hero-left"):hero.index("home-hero-right")]
    assert "home-pro-hero-cta" not in left


def test_right_column_stacks_card_over_cta():
    block = DASH_CSS[DASH_CSS.index(".home-hero-right {"):DASH_CSS.index(".home-hero-right {") + 220]
    assert "flex-direction: column" in block


def test_pricing_copy_matches_pricing_page():
    home = APP_PY[APP_PY.index('FORM_BODY = """'):]
    assert "Less than $1 a month" not in home
    assert "From $5 a year" in home


def test_weekly_recap_copy_scopes_pro_to_storyline():
    assert "Only the AI storyline is PRO" in APP_PY


def test_claim_team_modal_title():
    assert "Claim your team" in APP_PY
    assert "Sign in to your team" not in APP_PY


def test_no_em_dashes_in_new_copy():
    for snippet in ("From $5 a year", "Only the AI storyline is PRO", "Claim your team"):
        assert "\u2014" not in snippet
