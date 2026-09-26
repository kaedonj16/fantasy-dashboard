"""Plan restructure (UI worker, reconciled with backend core #1969).

Covers the UI half of the billing plan restructure:
- Pricing page shows exactly the 3 sellable plans (Starter / All-Pro / Hall of
  Fame) with league counts under the names, the monthly/annual toggle prices,
  "Save 44% with annual billing" callouts, and the "Recommended" badge on
  All-Pro. Retired plans (league, combo, single_league, old user) never appear
  in the purchase UI.
- static/paywall.js: BR_PRO_PLANS matches the catalog; the paywall modal and
  home PRO wizard render the new plans. No plan needs a league before checkout
  (BR_LEAGUE_PLANS is empty); the wizard has no league step, and no-league
  buyers land on /pricing after checkout so the Your PRO card can nudge them
  to assign their league slots.
- The "Your PRO" card renders Worker 1's slot picker (checkbox list wired to
  GET/POST /api/billing/pro-leagues) for slot plans with a persistent
  "N of M slots used" count, an unlimited-leagues note for Hall of Fame, and
  nothing for retired plans. Exactly one picker renders in the card.
- No em dashes in pricing UI copy.
"""
import re
from pathlib import Path

import pytest

pytest.importorskip("flask")

import routes.billing_bp as billing

ROOT = Path(__file__).resolve().parents[1]
PAYWALL_JS = (ROOT / "static" / "paywall.js").read_text(encoding="utf-8")


def _plan_cards_region(html: str) -> str:
    start = html.index("pricing-plan-grid")
    end = html.index("pricing-proof")
    return html[start:end]


def _cards(html: str):
    return re.findall(
        r'<article class="pricing-option[^"]*" data-plan-card="([^"]+)">(.*?)</article>',
        _plan_cards_region(html),
        re.S,
    )


# ── Pricing page ──────────────────────────────────────────────────────────────

def test_pricing_page_has_exactly_three_tiers(offline_client):
    html = offline_client.get("/pricing").get_data(as_text=True)
    cards = _cards(html)
    assert [key for key, _ in cards] == ["starter", "all_pro", "hall_of_fame"]


def test_pricing_page_league_count_under_each_name(offline_client):
    html = offline_client.get("/pricing").get_data(as_text=True)
    expected = {
        "starter": ("Starter", "1 league"),
        "all_pro": ("All-Pro", "5 leagues"),
        "hall_of_fame": ("Hall of Fame", "Unlimited leagues"),
    }
    for key, body in _cards(html):
        name, leagues = expected[key]
        assert f"<h3>{name}</h3>" in body
        assert f'<p class="pricing-leagues">{leagues}</p>' in body


def test_pricing_page_prices_and_savings(offline_client):
    html = offline_client.get("/pricing").get_data(as_text=True)
    expected = {
        "starter": ("$10<span>/year</span>", "$1.49<span>/mo</span>"),
        "all_pro": ("$30<span>/year</span>", "$4.49<span>/mo</span>"),
        "hall_of_fame": ("$50<span>/year</span>", "$7.49<span>/mo</span>"),
    }
    for key, body in _cards(html):
        annual, monthly = expected[key]
        assert annual in body, key
        assert monthly in body, key
        assert "Save 44% with annual billing" in body, key


def test_pricing_page_recommended_badge_on_all_pro(offline_client):
    html = offline_client.get("/pricing").get_data(as_text=True)
    cards = dict(_cards(html))
    assert "Recommended" in cards["all_pro"]
    assert "Recommended" not in cards["starter"]
    assert "Recommended" not in cards["hall_of_fame"]
    assert "Most popular" not in _plan_cards_region(html)


def test_pricing_page_no_retired_plans_in_purchase_ui(offline_client):
    html = offline_client.get("/pricing").get_data(as_text=True)
    region = _plan_cards_region(html)
    for retired in (
        "Individual: One League",
        "Entire League",
        "League + Personal",
        ">Personal<",
        "single_league",
    ):
        assert retired not in region, retired
    assert "Choose your PRO plan" in html


def test_pricing_savings_math():
    for plan in ("starter", "all_pro", "hall_of_fame"):
        assert billing._annual_savings_pct(plan) == 44


def test_pricing_body_has_no_em_dashes():
    body = billing.__doc__ or ""
    src = Path(billing.__file__).read_text(encoding="utf-8")
    chunk = src[src.index("def _pricing_body"):src.index("def page_league_pro_invite")]
    assert "\u2014" not in chunk
    assert "&mdash;" not in chunk


# ── PRO leagues slot picker ("Your PRO" card) ───────────────────────────────
# Worker 1's backend-integrated picker is the single implementation: a
# checkbox list wired to GET/POST /api/billing/pro-leagues, with a persistent
# "N of M slots used" count. Hall of Fame gets an unlimited-leagues note;
# retired plans get nothing.

def test_slot_caps_come_from_backend():
    assert billing.slot_cap_for_plan("starter") == 1
    assert billing.slot_cap_for_plan("all_pro") == 5
    assert billing.slot_cap_for_plan("hall_of_fame") is None


def test_slot_picker_js_contract(monkeypatch):
    # The slot-picker JS ships with the manage card (subscribers only).
    html = _manage_card_html(monkeypatch, "starter")
    assert "fetch('/api/billing/pro-leagues?platform='" in html
    assert "method: 'POST'" in html
    assert "brSaveProLeagues" in html
    assert "'/api/my-leagues'" in html
    assert "league_ids" in html
    # Persistent count + swap copy ported from the UI worker
    assert "slots used" in html
    assert "free a slot" in html
    assert "not in your saved leagues" in html


def _manage_card_html(monkeypatch, plan):
    from flask import Flask
    import utils.churn as churn

    monkeypatch.setattr(
        churn, "active_subscriptions_for_user",
        lambda user_id: [{
            "plan": plan,
            "league_id": "",
            "stripe_subscription_id": "sub_1",
            "expires_at": None,
        }],
    )
    app = Flask(__name__)
    app.secret_key = "test"
    with app.test_request_context("/pricing"):
        from flask import session as _session
        _session["viewer_user_id"] = "u1"
        return billing._pricing_manage_card()


def test_manage_card_renders_one_picker_for_starter_sub(monkeypatch):
    html = _manage_card_html(monkeypatch, "starter")
    assert "Starter PRO" in html
    # Exactly one picker: Worker 1's br-slots block. The duplicate
    # data-pro-leagues-mount implementation must not render.
    assert html.count('class="br-slots"') == 1
    assert 'class="br-slots-count"' in html
    assert "data-pro-leagues-mount" not in html


def test_manage_card_unlimited_note_for_hall_of_fame_sub(monkeypatch):
    html = _manage_card_html(monkeypatch, "hall_of_fame")
    assert "Hall of Fame PRO" in html
    assert 'class="br-slots-count"' not in html
    assert 'class="br-slots-list"' not in html
    assert 'onclick="brSaveProLeagues' not in html
    assert "data-pro-leagues-mount" not in html
    assert "unlimited leagues" in html.lower()


def test_manage_card_no_picker_for_retired_plans(monkeypatch):
    for plan in ("league", "combo", "single_league", "user"):
        html = _manage_card_html(monkeypatch, plan)
        assert 'class="br-slots"' not in html
        assert "data-pro-leagues-mount" not in html


# ── paywall.js ────────────────────────────────────────────────────────────────

def test_paywall_plan_catalog_matches():
    plans = PAYWALL_JS[PAYWALL_JS.index("const BR_PRO_PLANS"):PAYWALL_JS.index("function proPlanCards")]
    for key, name, annual, monthly in (
        ("starter", "Starter", "$10/year", "$1.49/mo"),
        ("all_pro", "All-Pro", "$30/year", "$4.49/mo"),
        ("hall_of_fame", "Hall of Fame", "$50/year", "$7.49/mo"),
    ):
        assert f"key: '{key}'" in plans, key
        assert f"name: '{name}'" in plans, key
        assert annual in plans, key
        assert monthly in plans, key
    assert "recommended: true" in plans
    assert "Recommended" in PAYWALL_JS[PAYWALL_JS.index("function proPlanCards"):PAYWALL_JS.index("window.showPaywall")]
    assert "pricing-leagues" in PAYWALL_JS


def test_paywall_no_league_required_at_checkout():
    # No plan requires a league before checkout: Starter/All-Pro buyers pick
    # their slot leagues after purchase from the Your PRO card on /pricing.
    assert "const BR_LEAGUE_PLANS = {};" in PAYWALL_JS
    assert "{ starter: true" not in PAYWALL_JS
    # No-league buyers land on /pricing after checkout to assign slots.
    assert "/pricing?new_subscriber=1&welcome=${_welcome}" in PAYWALL_JS


def test_paywall_no_old_plan_keys_in_purchase_flow():
    purchase = PAYWALL_JS[PAYWALL_JS.index("async function initiatePurchase"):]
    purchase = purchase[:PAYWALL_JS.index("function addPremiumBadge") - PAYWALL_JS.index("async function initiatePurchase")]
    for old in ("'league'", "'combo'", "'single_league'"):
        assert old not in purchase, old


def test_home_wizard_has_no_league_step():
    wizard = PAYWALL_JS[PAYWALL_JS.index("function openHomeProModal"):PAYWALL_JS.index("window.openHomeProModal")]
    # The league step is gone: picking a plan goes straight to Google/checkout.
    assert "homeProStepLeague" not in wizard
    assert "BR_LEAGUE_PLANS" not in wizard
    assert "proPlanCards({ dataPlan: true })" in wizard
    assert "initiatePurchase(selectedPlan, btn)" in wizard
    assert "_startGoogleSubscribe(selectedPlan, btn)" in wizard
    assert "Your PRO card" in wizard


def test_paywall_js_has_no_em_dashes():
    assert "\u2014" not in PAYWALL_JS
    assert "&mdash;" not in PAYWALL_JS
