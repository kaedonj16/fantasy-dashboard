"""Subscription/terms copy: auto-renewal, cancellation, refunds on Terms and
pricing; NFL non-affiliation; guide corrections."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
PUBLIC_BP = (ROOT / "routes" / "public_bp.py").read_text(encoding="utf-8")
BILLING = (ROOT / "routes" / "billing_bp.py").read_text(encoding="utf-8")
GUIDES = (ROOT / "routes" / "guides_content.py").read_text(encoding="utf-8")


def test_terms_has_subscription_section():
    assert "Subscriptions" in PUBLIC_BP
    assert "renews" in PUBLIC_BP and "automatically each year" in PUBLIC_BP
    assert "cancel anytime" in PUBLIC_BP
    assert "non-refundable" in PUBLIC_BP


def test_terms_updated_date():
    assert "Last updated: September 25, 2026" in PUBLIC_BP


def test_nfl_disclaimer():
    assert "Not affiliated with the NFL" in APP_PY
    assert "not affiliated with, and do not endorse, BR Fantasy" in PUBLIC_BP


def test_pricing_faq_covers_billing_renewal_cancel_refund():
    assert "How does PRO billing work?" in BILLING
    assert "How do I cancel?" in BILLING
    assert "Can I get a refund?" in BILLING
    assert "non-refundable" in BILLING


def test_te_premium_guide_17_game_era():
    assert "Over 16 games" not in GUIDES
    assert "17-game season" in GUIDES


def test_flagship_guide_lede_typo_fixed():
    assert "contract of expected production" not in GUIDES
    assert "age, contract, expected production" in GUIDES


def test_no_em_dashes_in_new_copy():
    for snippet in (
        "renews automatically each year",
        "Not affiliated with the NFL",
        "How does PRO billing work?",
        "17-game season",
    ):
        assert "\u2014" not in snippet
