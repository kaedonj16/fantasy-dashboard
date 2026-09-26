"""Plan restructure: Starter / All-Pro / Hall of Fame catalog.

Covers the billing-plan restructure without touching live Stripe or a real
database (fake cursors + offline_client):

- Catalog constants: three purchasable plans, four grandfathered legacy keys,
  six new Stripe Price env vars, slot caps 1/5/unlimited.
- slot_cap_for_plan unit behavior.
- Migration 039 adds plan_key and the normalized pro_league_slots table.
- Price-ID inference for dashboard Price subscriptions in webhooks.
- Slot-capped entitlement: starter/all_pro grant only selected leagues;
  hall_of_fame, grandfathered 'user', and legacy NULL rows stay unlimited.
- GET/POST /api/billing/pro-leagues: session identity only, cap enforced,
  duplicates normalized, foreign leagues rejected.
- Legacy plans rejected for new checkout; legacy resumed checkout and
  legacy webhook renewals keep working.
- Lifecycle sync updates plan_key on plan changes.
- Win-back sells hall_of_fame (the unlimited-plan successor).
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("flask")

import routes.billing_bp as billing
from dashboard_services import subscriptions

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime.now(timezone.utc)
FUTURE = NOW + timedelta(days=30)


# ── Catalog constants ─────────────────────────────────────────────────────────

def test_current_catalog_amounts():
    assert {k: v["unit_amount"] for k, v in billing._STRIPE_PRICES.items()} == {
        "starter": 1000,
        "all_pro": 3000,
        "hall_of_fame": 5000,
    }


def test_monthly_defaults():
    assert billing._MONTHLY_DEFAULT_UNIT_AMOUNTS == {
        "starter": 149,
        "all_pro": 449,
        "hall_of_fame": 749,
    }


def test_six_new_price_env_names():
    assert billing._ANNUAL_PRICE_ENV == {
        "starter": "STRIPE_PRICE_STARTER_ANNUAL",
        "all_pro": "STRIPE_PRICE_ALL_PRO_ANNUAL",
        "hall_of_fame": "STRIPE_PRICE_HALL_OF_FAME_ANNUAL",
    }
    assert billing._MONTHLY_PRICE_ENV == {
        "starter": "STRIPE_PRICE_STARTER_MONTHLY",
        "all_pro": "STRIPE_PRICE_ALL_PRO_MONTHLY",
        "hall_of_fame": "STRIPE_PRICE_HALL_OF_FAME_MONTHLY",
    }


def test_known_plans_includes_legacy_but_not_purchasable():
    assert billing._KNOWN_PLANS == {
        "starter", "all_pro", "hall_of_fame",
        "league", "user", "combo", "single_league",
    }
    assert set(billing._STRIPE_PRICES) == {"starter", "all_pro", "hall_of_fame"}
    assert set(billing._LEGACY_PRICES) == {"league", "user", "combo", "single_league"}


def test_plan_league_caps():
    assert billing._PLAN_LEAGUE_CAP["starter"] == 1
    assert billing._PLAN_LEAGUE_CAP["all_pro"] == 5
    assert billing._PLAN_LEAGUE_CAP.get("hall_of_fame") is None


def test_slot_cap_for_plan():
    assert subscriptions.slot_cap_for_plan("starter") == 1
    assert subscriptions.slot_cap_for_plan("all_pro") == 5
    assert subscriptions.slot_cap_for_plan("hall_of_fame") is None
    assert subscriptions.slot_cap_for_plan("user") is None
    assert subscriptions.slot_cap_for_plan("") is None
    assert subscriptions.slot_cap_for_plan(None) is None
    assert subscriptions.slot_cap_for_plan("bogus") is None
    assert subscriptions.slot_cap_for_plan(" STARTER ") == 1


# ── Migration 039 ─────────────────────────────────────────────────────────────

def test_migration_039_adds_plan_key_and_slots_table():
    sql = (ROOT / "migrations" / "039_plan_restructure.sql").read_text(encoding="utf-8")
    assert "plan_key" in sql
    assert "ADD COLUMN IF NOT EXISTS" in sql
    assert "CREATE TABLE IF NOT EXISTS pro_league_slots" in sql
    assert "UNIQUE" in sql
    assert "user_id" in sql and "league_id" in sql and "platform" in sql
    # Slots stay valid while the owning subscription row is active (enforced by
    # the entitlement JOIN); the table itself carries no independent expiry.
    assert "stripe_subscription_id" in sql
    for index in ("idx_pro_league_slots_user", "idx_pro_league_slots_league"):
        assert index in sql


# ── Price-ID inference ────────────────────────────────────────────────────────

def test_plan_from_subscription_prefers_price_id(monkeypatch):
    monkeypatch.setenv("STRIPE_PRICE_ALL_PRO_ANNUAL", "price_all_pro_123")
    sub = {
        "items": {"data": [{
            "price": {"id": "price_all_pro_123", "product": "prod_unknown"},
        }]},
    }
    assert billing._plan_from_subscription(sub) == "all_pro"


def test_plan_from_subscription_legacy_product_still_works(monkeypatch):
    monkeypatch.setattr(billing, "_STRIPE_USER_PRODUCT", "prod_user_legacy")
    sub = {
        "items": {"data": [{
            "price": {"id": "price_other", "product": "prod_user_legacy"},
        }]},
    }
    assert billing._plan_from_subscription(sub) == "user"


def test_plan_from_subscription_unknown_returns_blank():
    sub = {"items": {"data": [{"price": {"id": "price_x", "product": "prod_x"}}]}}
    assert billing._plan_from_subscription(sub) == ""


# ── Fake cursor for entitlement checks ────────────────────────────────────────

class _PlanCursor:
    """Emulates the user_subscriptions + pro_league_slots + pro_trials reads."""

    def __init__(self, personal_row=None, slots=()):
        self.personal_row = personal_row
        self.slots = set(slots)
        self.queries = []
        self._kind = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def execute(self, sql, params=None):
        self.queries.append((sql, params))
        if "FROM pro_league_slots pls" in sql:
            self._kind = ("slot", params[2], params[4])
        elif "FROM user_subscriptions" in sql and "plan_key" in sql:
            self._kind = ("personal",)
        else:
            self._kind = ("other",)

    def fetchone(self):
        kind = self._kind
        if kind[0] == "personal":
            return dict(self.personal_row) if self.personal_row else None
        if kind[0] == "slot":
            _, user_key, league_id = kind
            row = self.personal_row or {}
            pk = (row.get("plan_key") or "").strip().lower()
            if (user_key, league_id) in self.slots and pk in ("starter", "all_pro"):
                return {"1": 1}
            return None
        return None

    def fetchall(self):
        return []


class _PlanConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def cursor(self):
        return self._cursor


def _install_plan_conn(monkeypatch, cursor):
    monkeypatch.setattr(subscriptions, "get_conn", lambda: _PlanConn(cursor))


def _personal_row(plan_key, **over):
    row = {
        "expires_at": FUTURE,
        "stripe_customer_id": "cus_1",
        "billing_interval": "year",
        "plan_key": plan_key,
    }
    row.update(over)
    return row


def _entitlement(monkeypatch, user_id, league_id, personal_row, slots=()):
    cursor = _PlanCursor(personal_row=personal_row, slots=slots)
    _install_plan_conn(monkeypatch, cursor)
    return subscriptions.has_premium_access(user_id, league_id, "sleeper")


# ── Slot-capped entitlement ───────────────────────────────────────────────────

def test_starter_grants_only_selected_league(monkeypatch):
    assert _entitlement(monkeypatch, "u1", "lgA", _personal_row("starter"), slots={("u1", "lgA")}) is True
    assert _entitlement(monkeypatch, "u1", "lgB", _personal_row("starter"), slots={("u1", "lgA")}) is False


def test_starter_with_no_slots_grants_nothing(monkeypatch):
    assert _entitlement(monkeypatch, "u1", "lgA", _personal_row("starter"), slots=set()) is False


def test_all_pro_grants_up_to_five_slots(monkeypatch):
    slots = {("u1", f"lg{i}") for i in range(5)}
    assert _entitlement(monkeypatch, "u1", "lg3", _personal_row("all_pro"), slots=slots) is True
    assert _entitlement(monkeypatch, "u1", "lgX", _personal_row("all_pro"), slots=slots) is False


def test_hall_of_fame_is_unlimited(monkeypatch):
    row = _personal_row("hall_of_fame")
    assert _entitlement(monkeypatch, "u1", "lgAny", row) is True
    assert _entitlement(monkeypatch, "u1", "lgOther", row) is True


def test_grandfathered_user_plan_stays_unlimited(monkeypatch):
    row = _personal_row("user")
    assert _entitlement(monkeypatch, "u1", "lgAny", row) is True
    assert _entitlement(monkeypatch, "u1", "lgOther", row) is True


def test_legacy_null_plan_key_reads_as_unlimited_user(monkeypatch):
    row = _personal_row(None)
    assert _entitlement(monkeypatch, "u1", "lgAny", row) is True


def test_no_subscription_grants_nothing(monkeypatch):
    assert _entitlement(monkeypatch, "u1", "lgA", None) is False


# ── Manage card slot picker ───────────────────────────────────────────────────

def _starter_sub(plan="starter", platform="sleeper"):
    return [{
        "table": "user_subscriptions", "stripe_subscription_id": "sub_1",
        "plan": plan, "expires_at": FUTURE, "league_id": "",
        "platform": platform,
    }]


def test_manage_card_shows_slot_picker_for_starter(offline_client, monkeypatch):
    _signed_in(offline_client)
    monkeypatch.setattr(
        "utils.churn.active_subscriptions_for_user",
        lambda uid: _starter_sub("starter"),
    )
    resp = offline_client.get("/pricing")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert 'class="br-slots"' in html
    assert 'data-platform="sleeper"' in html
    assert "brSaveProLeagues" in html
    assert "1 league" in html


def test_manage_card_no_picker_for_unlimited_plan(offline_client, monkeypatch):
    _signed_in(offline_client)
    monkeypatch.setattr(
        "utils.churn.active_subscriptions_for_user",
        lambda uid: _starter_sub("hall_of_fame"),
    )
    resp = offline_client.get("/pricing")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Your PRO" in html
    assert 'class="br-slots"' not in html


# ── Legacy single-league duplicate detection covers all aliases ───────────────

def test_legacy_single_check_covers_all_aliases(monkeypatch):
    seen = []

    class _AnyCursor:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def execute(self, sql, params=None):
            seen.append((sql, params))
        def fetchone(self): return None
        def fetchall(self): return []

    class _AnyConn:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def cursor(self): return _AnyCursor()

    monkeypatch.setattr(subscriptions, "get_conn", lambda: _AnyConn())
    assert subscriptions.has_any_user_league_subscription(
        "sleeper-9", "sleeper", account_id=42,
        user_keys=["sleeper-9", "usernine"],
    ) is False
    single_checks = [
        params for sql, params in seen
        if sql.strip().startswith("SELECT 1 FROM user_league_subscriptions")
    ]
    # First: direct identities (checkout id + username); then acct keys.
    assert single_checks[0][0] == ["sleeper-9", "usernine"]
    assert single_checks[1][0] == ["acct:42", "42"]


# ── _ensure_plan_key memoization ──────────────────────────────────────────────

def test_ensure_plan_key_runs_ddl_once_per_process(monkeypatch):
    cursor = _PlanCursor()
    monkeypatch.setattr(subscriptions, "_PLAN_KEY_ENSURED", False)
    subscriptions._ensure_plan_key(cursor)
    subscriptions._ensure_plan_key(cursor)
    alters = [sql for sql, _ in cursor.queries if "ADD COLUMN IF NOT EXISTS plan_key" in sql]
    assert len(alters) == 1


# ── Slot helpers: cap + dedup ─────────────────────────────────────────────────

def _with_active_plan(monkeypatch, plan_key):
    monkeypatch.setattr(
        subscriptions, "find_active_personal_plan",
        lambda user_keys, platform="sleeper": {
            "user_key": user_keys[0], "plan_key": plan_key,
            "expires_at": FUTURE, "stripe_subscription_id": "sub_1",
        },
    )


def test_set_pro_league_slots_enforces_cap(monkeypatch):
    cursor = _PlanCursor(personal_row=_personal_row("starter"))
    _install_plan_conn(monkeypatch, cursor)
    _with_active_plan(monkeypatch, "starter")
    with pytest.raises(ValueError, match="covers 1 league"):
        subscriptions.set_pro_league_slots("u1", "sleeper", ["lgA", "lgB"])


def test_set_pro_league_slots_dedupes(monkeypatch):
    cursor = _PlanCursor(personal_row=_personal_row("all_pro"))
    _install_plan_conn(monkeypatch, cursor)
    _with_active_plan(monkeypatch, "all_pro")
    saved = subscriptions.set_pro_league_slots(
        "u1", "sleeper", ["lgA", "lgA", " lgB "],
    )
    assert saved == ["lgA", "lgB"]
    inserts = [p for sql, p in cursor.queries if "INSERT INTO pro_league_slots" in sql]
    assert len(inserts) == 1
    assert inserts[0][-1] == ["lgA", "lgB"]


def test_set_pro_league_slots_replaces_old_set(monkeypatch):
    cursor = _PlanCursor(personal_row=_personal_row("all_pro"))
    _install_plan_conn(monkeypatch, cursor)
    _with_active_plan(monkeypatch, "all_pro")
    subscriptions.set_pro_league_slots("u1", "sleeper", ["lgA"])
    deletes = [sql for sql, _ in cursor.queries if sql.strip().startswith("DELETE FROM pro_league_slots")]
    assert len(deletes) == 1


# ── Slot endpoints ────────────────────────────────────────────────────────────

def _signed_in(offline_client, **extra):
    with offline_client.session_transaction() as sess:
        sess["account_id"] = 99
        sess["viewer_user_id"] = "sleeper-99"
        for k, v in extra.items():
            sess[k] = v


def test_pro_leagues_get_requires_google(offline_client):
    assert offline_client.get("/api/billing/pro-leagues").status_code == 401


def test_pro_leagues_get_404_without_plan(offline_client, monkeypatch):
    _signed_in(offline_client)
    monkeypatch.setattr(billing, "find_active_personal_plan", lambda keys, platform="sleeper": None)
    assert offline_client.get("/api/billing/pro-leagues").status_code == 404


def test_pro_leagues_get_returns_plan_cap_and_slots(offline_client, monkeypatch):
    _signed_in(offline_client)
    monkeypatch.setattr(
        billing, "find_active_personal_plan",
        lambda keys, platform="sleeper": {
            "user_key": "sleeper-99", "plan_key": "all_pro",
            "expires_at": FUTURE, "stripe_subscription_id": "sub_1",
        },
    )
    monkeypatch.setattr(billing, "get_pro_league_slots", lambda uk, platform: ["lgA", "lgB"])
    resp = offline_client.get("/api/billing/pro-leagues")
    assert resp.status_code == 200
    assert resp.get_json() == {"plan": "all_pro", "cap": 5, "league_ids": ["lgA", "lgB"]}


def test_pro_leagues_get_unlimited_plan_needs_no_slots(offline_client, monkeypatch):
    _signed_in(offline_client)
    monkeypatch.setattr(
        billing, "find_active_personal_plan",
        lambda keys, platform="sleeper": {
            "user_key": "sleeper-99", "plan_key": "hall_of_fame",
            "expires_at": FUTURE, "stripe_subscription_id": "sub_1",
        },
    )
    resp = offline_client.get("/api/billing/pro-leagues")
    assert resp.get_json() == {"plan": "hall_of_fame", "cap": None, "league_ids": []}


def test_pro_leagues_post_requires_google(offline_client):
    assert offline_client.post(
        "/api/billing/pro-leagues", json={"league_ids": ["lgA"]}
    ).status_code == 401


def test_pro_leagues_post_rejects_non_list(offline_client, monkeypatch):
    _signed_in(offline_client)
    monkeypatch.setattr(
        billing, "find_active_personal_plan",
        lambda keys, platform="sleeper": {"user_key": "sleeper-99", "plan_key": "starter"},
    )
    resp = offline_client.post("/api/billing/pro-leagues", json={"league_ids": "lgA"})
    assert resp.status_code == 400


def test_pro_leagues_post_enforces_cap(offline_client, monkeypatch):
    _signed_in(offline_client)
    monkeypatch.setattr(
        billing, "find_active_personal_plan",
        lambda keys, platform="sleeper": {
            "user_key": "sleeper-99", "plan_key": "starter",
            "stripe_subscription_id": "sub_1",
        },
    )
    resp = offline_client.post(
        "/api/billing/pro-leagues", json={"league_ids": ["lgA", "lgB"]},
    )
    assert resp.status_code == 400
    assert "1 league" in resp.get_json()["error"]


def test_pro_leagues_post_dedupes_and_replaces(offline_client, monkeypatch):
    _signed_in(offline_client)
    monkeypatch.setattr(
        billing, "find_active_personal_plan",
        lambda keys, platform="sleeper": {
            "user_key": "sleeper-99", "plan_key": "all_pro",
            "stripe_subscription_id": "sub_1",
        },
    )
    seen = {}

    def fake_set(user_key, platform, league_ids, stripe_subscription_id=None):
        seen.update(
            user_key=user_key, platform=platform, league_ids=league_ids,
            stripe_subscription_id=stripe_subscription_id,
        )
        return list(dict.fromkeys(league_ids))

    monkeypatch.setattr(billing, "set_pro_league_slots", fake_set)
    monkeypatch.setattr(
        "dashboard_services.subscriptions.viewer_is_league_member",
        lambda *a, **k: True,
    )
    resp = offline_client.post(
        "/api/billing/pro-leagues", json={"league_ids": ["lgA", "lgA", "lgB"]},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["league_ids"] == ["lgA", "lgB"]
    assert body["cap"] == 5
    assert seen["user_key"] == "sleeper-99"
    assert seen["stripe_subscription_id"] == "sub_1"


def test_pro_leagues_post_rejects_foreign_league(offline_client, monkeypatch):
    _signed_in(offline_client)
    monkeypatch.setattr(
        billing, "find_active_personal_plan",
        lambda keys, platform="sleeper": {
            "user_key": "sleeper-99", "plan_key": "all_pro",
            "stripe_subscription_id": "sub_1",
        },
    )
    monkeypatch.setattr(
        billing, "set_pro_league_slots",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not write")),
    )
    monkeypatch.setattr(
        "dashboard_services.subscriptions.viewer_is_league_member",
        lambda member_id, league_id, platform, season: league_id == "lgMine",
    )
    resp = offline_client.post(
        "/api/billing/pro-leagues", json={"league_ids": ["lgMine", "lgTheirs"]},
    )
    assert resp.status_code == 400
    assert "not on your account" in resp.get_json()["error"]


def test_pro_leagues_post_rejects_unlimited_plan(offline_client, monkeypatch):
    _signed_in(offline_client)
    monkeypatch.setattr(
        billing, "find_active_personal_plan",
        lambda keys, platform="sleeper": {
            "user_key": "sleeper-99", "plan_key": "hall_of_fame",
            "stripe_subscription_id": "sub_1",
        },
    )
    resp = offline_client.post(
        "/api/billing/pro-leagues", json={"league_ids": ["lgA"]},
    )
    assert resp.status_code == 400


# ── Legacy checkout rejection / resume ────────────────────────────────────────

def test_new_checkout_rejects_legacy_plans(offline_client, monkeypatch):
    monkeypatch.setattr(
        billing, "_stripe",
        lambda: (_ for _ in ()).throw(AssertionError("Stripe must not be called")),
    )
    with offline_client.session_transaction() as sess:
        sess["account_id"] = 5
        sess["viewer_user_id"] = "sleeper-5"
    for plan in ("league", "user", "combo", "single_league"):
        resp = offline_client.post("/api/create-checkout-session", json={
            "plan": plan, "platform": "sleeper", "season": 2026,
        })
        assert resp.status_code == 400, plan
        assert "invalid plan" in resp.get_json()["error"].lower()


def test_stripe_checkout_url_rejects_legacy_without_flag(monkeypatch):
    from flask import Flask

    tiny = Flask(__name__)
    with tiny.test_request_context("https://example.com/pricing"):
        url, err = billing._stripe_checkout_url(
            "acct:1", {"plan": "user", "platform": "sleeper", "season": 2026},
        )
    assert url is None
    assert err == "Invalid plan"


def test_legacy_resume_checkout_still_opens_stripe(offline_client, monkeypatch):
    captured = {}

    class _CheckoutSession:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(url="https://checkout.stripe.test/legacy")

    monkeypatch.setattr(
        billing, "_stripe",
        lambda: SimpleNamespace(checkout=SimpleNamespace(Session=_CheckoutSession)),
    )
    monkeypatch.setattr(
        "dashboard_services.subscriptions.viewer_is_league_member",
        lambda *a, **k: True,
    )
    with offline_client.session_transaction() as sess:
        sess["account_id"] = 6
        sess["viewer_user_id"] = "sleeper-6"
        # Session staged before the cutover.
        sess["pending_checkout"] = {
            "plan": "user", "league_id": "", "platform": "sleeper", "season": 2026,
        }
    resp = offline_client.get("/pro/resume-checkout")
    assert resp.status_code == 302
    assert resp.headers["Location"] == "https://checkout.stripe.test/legacy"
    assert captured["metadata"]["plan"] == "user"


# ── Legacy webhook renewal ────────────────────────────────────────────────────

def _webhook_stripe(monkeypatch, event, sub=None):
    fake = SimpleNamespace(
        Webhook=SimpleNamespace(construct_event=lambda *a, **k: event),
        SignatureVerificationError=type("SigErr", (Exception,), {}),
        Subscription=SimpleNamespace(
            retrieve=lambda sid: sub or SimpleNamespace(current_period_end=1_900_000_000),
        ),
    )
    monkeypatch.setattr(billing, "_stripe", lambda: fake)
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")


class _Meta(dict):
    def to_dict(self):
        return dict(self)


def test_legacy_user_renewal_grant_keeps_plan_key(offline_client, monkeypatch):
    captured = {}

    def fake_create(user_id, expires_at, **kwargs):
        captured.update(kwargs)
        captured["user_id"] = user_id
        return True

    monkeypatch.setattr(billing, "create_user_subscription", fake_create)
    monkeypatch.setattr(billing, "_STRIPE_USER_PRODUCT", "prod_user_legacy")
    session = SimpleNamespace(
        metadata=_Meta(plan="user", user_id="u1", platform="sleeper", interval="year"),
        subscription="sub_legacy",
        customer="cus_legacy",
    )
    sub = {
        "items": {"data": [{
            "price": {
                "id": "price_old", "product": "prod_user_legacy",
                "recurring": {"interval": "year"},
            },
        }]},
        "current_period_end": 1_900_000_000,
    }
    _webhook_stripe(
        monkeypatch,
        {"type": "checkout.session.completed", "data": {"object": session}},
        sub=sub,
    )
    resp = offline_client.post(
        "/api/stripe-webhook", data=b"{}",
        headers={"Stripe-Signature": "t"},
    )
    assert resp.status_code == 200
    assert captured["user_id"] == "u1"
    assert captured["plan_key"] == "user"


# ── Lifecycle plan_key sync on plan changes ───────────────────────────────────

class _LifecycleCursor:
    def __init__(self):
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def execute(self, sql, params=None):
        self.queries.append((sql, params))

    def fetchone(self):
        for sql, _ in self.queries[-1:]:
            if "SELECT id FROM user_subscriptions" in sql:
                return {"id": 3}
        return None

    def fetchall(self):
        return []


class _LifecycleConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def cursor(self):
        return self._cursor


def test_lifecycle_plan_change_updates_plan_key(monkeypatch):
    cursor = _LifecycleCursor()
    monkeypatch.setattr(subscriptions, "get_conn", lambda: _LifecycleConn(cursor))
    monkeypatch.setattr(subscriptions, "_BILLING_INTERVAL_ENSURED", set())
    monkeypatch.setattr(subscriptions, "_PLAN_KEY_ENSURED", False)
    summary = subscriptions.apply_subscription_lifecycle(
        "sub_1",
        event_type="customer.subscription.updated",
        status="active",
        cancel_at_period_end=False,
        expires_at=FUTURE,
        plan="all_pro",
        interval="year",
        user_id="u1",
        platform="sleeper",
        plan_key="all_pro",
    )
    assert "user_subscriptions" in summary["updated"]
    updates = [
        (sql, params) for sql, params in cursor.queries
        if sql.startswith("UPDATE user_subscriptions")
    ]
    assert updates, "expected a user_subscriptions UPDATE"
    assert "plan_key" in updates[0][0]
    assert "all_pro" in updates[0][1]


# ── Win-back sells hall_of_fame ───────────────────────────────────────────────

def test_winback_checkout_uses_hall_of_fame(offline_client, monkeypatch):
    captured = {}

    class _CheckoutSession:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(url="https://checkout.stripe.test/winback")

    monkeypatch.setattr(
        billing, "_stripe",
        lambda: SimpleNamespace(checkout=SimpleNamespace(Session=_CheckoutSession)),
    )
    monkeypatch.setattr("utils.churn.verify_winback_token", lambda token: 77)
    monkeypatch.setattr("utils.churn.account_has_active_sub", lambda aid: False)
    monkeypatch.setattr("utils.churn.winback_coupon_id", lambda: "coupon_20off")
    monkeypatch.setattr("utils.churn.record_event", lambda *a, **k: None)

    resp = offline_client.get("/pro/winback?token=tok123")
    assert resp.status_code == 302
    assert resp.headers["Location"] == "https://checkout.stripe.test/winback"
    assert captured["metadata"]["plan"] == "hall_of_fame"
    assert captured["discounts"] == [{"coupon": "coupon_20off"}]
    line = captured["line_items"][0]
    if "price" in line:
        assert line["quantity"] == 1
    else:
        assert line["price_data"]["unit_amount"] == 5000
        assert line["price_data"]["recurring"] == {"interval": "year"}
