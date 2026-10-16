"""Assistant tier resolution: the plan lookup must use the caller's full
identity key set, like every other entitlement read on the site.

Regression: a Hall of Fame account showed as Free in the assistant. The
plan row was stored under the owner's Sleeper identity (checkout stamps
grants with the session's viewer id / username first; the admin grant
script's default form keys by Sleeper id too), while the assistant only
looked up acct:<id> / the bare account id. The canonical gate
(subscriptions.has_premium_access) finds such rows by joining
user_subscriptions through account_identities.

store/subscriptions are faked in memory; no DB, no OpenAI.
"""
import pytest

# The AI client imports `openai`, which the lint job doesn't install; skip at
# collection there (same pattern as tests/test_assistant_service.py).
pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services import subscriptions
from dashboard_services.assistant import service


@pytest.fixture()
def billing(monkeypatch):
    """In-memory subscription store behind the real subscriptions API shape.

    personal: {(user_id, platform): plan_key} -- plan_key None models a
    legacy row whose plan_key is NULL (reads back as grandfathered 'user').
    single_league_keys: user ids holding an active user_league_subscriptions
    row. trial_keys: keys holding an active PRO trial.
    """
    state = {"personal": {}, "single_league_keys": set(), "trial_keys": set(),
             "plan_calls": []}

    def fake_find(keys, platform="sleeper"):
        state["plan_calls"].append((list(keys), platform))
        for key in keys:
            if (key, platform) in state["personal"]:
                pk = state["personal"][(key, platform)]
                return {"user_key": key, "plan_key": pk or "user",
                        "expires_at": None, "stripe_subscription_id": None}
        return None

    def fake_single(user_id, platform="sleeper", account_id=None, user_keys=None):
        keys = [user_id, *(user_keys or [])]
        return any(k and k in state["single_league_keys"] for k in keys)

    def fake_trial(keys):
        return any(k in state["trial_keys"] for k in keys)

    monkeypatch.setattr(subscriptions, "find_active_personal_plan", fake_find)
    monkeypatch.setattr(subscriptions, "has_any_user_league_subscription", fake_single)
    monkeypatch.setattr(subscriptions, "trial_active_for_keys", fake_trial)
    monkeypatch.setattr(service.store, "count_monthly_asked", lambda key: 0)
    monkeypatch.setattr(service, "_linked_identity_keys", lambda aid: ([], []))
    for env in ("ASSISTANT_EXEMPT_ACCOUNT_KEYS", "ASSISTANT_EXEMPT_EMAILS",
                "ASSISTANT_FREE_MONTHLY_CAP", "ASSISTANT_PAID_MONTHLY_CAP",
                "ASSISTANT_STARTER_MONTHLY_CAP", "ASSISTANT_ALL_PRO_MONTHLY_CAP",
                "ASSISTANT_HALL_OF_FAME_MONTHLY_CAP"):
        monkeypatch.delenv(env, raising=False)
    return state


def _link(monkeypatch, keys, platforms=("sleeper",)):
    monkeypatch.setattr(service, "_linked_identity_keys",
                        lambda aid: (list(keys), list(platforms)))


# -- where the row is keyed ----------------------------------------------------

def test_plan_keyed_by_account_key(billing):
    billing["personal"][("acct:42", "sleeper")] = "hall_of_fame"
    caps = service.caps_for("acct:42")
    assert caps["tier"] == "hall_of_fame" and caps["plan_key"] == "hall_of_fame"
    assert caps["limit"] == 1000


def test_plan_keyed_by_linked_sleeper_id(billing, monkeypatch):
    # The reported bug: the HOF row sits under the owner's Sleeper user id,
    # reachable only through account_identities.
    _link(monkeypatch, ["998877", "hoodiekj1"])
    billing["personal"][("998877", "sleeper")] = "hall_of_fame"
    caps = service.caps_for("acct:42")
    assert caps["tier"] == "hall_of_fame" and caps["plan_key"] == "hall_of_fame"


def test_plan_keyed_by_linked_handle(billing, monkeypatch):
    _link(monkeypatch, ["998877", "hoodiekj1"])
    billing["personal"][("hoodiekj1", "sleeper")] = "user"
    caps = service.caps_for("acct:42")
    assert caps["tier"] == "hall_of_fame" and caps["plan_key"] == "user"


def test_plan_keyed_by_session_viewer_identity(billing):
    from flask import Flask, session

    billing["personal"][("998877", "sleeper")] = "all_pro"
    app = Flask(__name__)
    app.secret_key = "test"
    with app.test_request_context("/"):
        session["account_id"] = 42
        session["viewer_user_id"] = "998877"
        session["viewer_username"] = "hoodiekj1"
        plan = service._active_plan("acct:42")
    assert plan is not None and plan["plan_key"] == "all_pro"


# -- every stored plan_key maps to its decided tier ----------------------------

@pytest.mark.parametrize("plan_key,tier,limit", [
    ("starter", "starter", 100),
    ("single_league", "starter", 100),
    ("all_pro", "all_pro", 250),
    ("league", "all_pro", 250),
    ("hall_of_fame", "hall_of_fame", 1000),
    ("user", "hall_of_fame", 1000),
    ("combo", "hall_of_fame", 1000),
    (None, "hall_of_fame", 1000),        # NULL plan_key = grandfathered 'user'
    ("some_future_plan", "starter", 100)  # unknown paid keys land on starter
])
def test_plan_key_mapping(billing, monkeypatch, plan_key, tier, limit):
    _link(monkeypatch, ["998877"])
    billing["personal"][("998877", "sleeper")] = plan_key
    caps = service.caps_for("acct:42")
    assert caps["tier"] == tier and caps["limit"] == limit
    assert caps["tier_label"] == service.TIER_LABELS[tier]


# -- representations outside user_subscriptions --------------------------------

def test_single_league_row_folds_to_starter(billing, monkeypatch):
    # Grandfathered single_league rows live in user_league_subscriptions;
    # the personal-plan lookup can never see them.
    _link(monkeypatch, ["998877"])
    billing["single_league_keys"].add("998877")
    caps = service.caps_for("acct:42")
    assert caps["tier"] == "starter" and caps["plan_key"] == "single_league"
    assert caps["limit"] == 100


def test_active_trial_is_full_pro(billing):
    billing["trial_keys"].add("acct:42")
    caps = service.caps_for("acct:42")
    assert caps["tier"] == "hall_of_fame" and caps["plan_key"] == "trial"


def test_paid_plan_wins_over_trial(billing):
    billing["personal"][("acct:42", "sleeper")] = "starter"
    billing["trial_keys"].add("acct:42")
    caps = service.caps_for("acct:42")
    assert caps["tier"] == "starter" and caps["plan_key"] == "starter"


def test_true_free(billing):
    caps = service.caps_for("acct:42")
    assert caps["tier"] == "free" and caps["plan_key"] is None
    assert caps["limit"] == 25


def test_best_tier_across_platforms(billing, monkeypatch):
    # The canonical account check spans platforms: a hall_of_fame row on a
    # linked Yahoo identity beats a starter row on Sleeper.
    _link(monkeypatch, ["998877", "yahoo-uid"], platforms=("sleeper", "yahoo"))
    billing["personal"][("acct:42", "sleeper")] = "starter"
    billing["personal"][("yahoo-uid", "yahoo")] = "hall_of_fame"
    caps = service.caps_for("acct:42")
    assert caps["tier"] == "hall_of_fame" and caps["plan_key"] == "hall_of_fame"


def test_lookup_failure_defaults_to_free(billing, monkeypatch):
    def boom(keys, platform="sleeper"):
        raise RuntimeError("db down")

    monkeypatch.setattr(subscriptions, "find_active_personal_plan", boom)
    caps = service.caps_for("acct:42")
    assert caps["tier"] == "free" and caps["plan_key"] is None


# -- the account_identities read itself -----------------------------------------

class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=()):
        pass

    def fetchall(self):
        return list(self._rows)


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return _FakeCursor(self._rows)


def test_linked_identity_keys_reads_account_identities(monkeypatch):
    rows = [
        {"platform": "sleeper", "platform_user_id": "998877", "handle": "hoodiekj1"},
        ("sleeper", "998877", "hoodiekj1"),  # tuple row, duplicate identity
        {"platform": "yahoo", "platform_user_id": "y-1", "handle": None},
    ]
    monkeypatch.setattr(subscriptions, "get_conn", lambda *a, **k: _FakeConn(rows))
    keys, platforms = subscriptions.linked_identity_keys(42)
    assert keys == ["998877", "hoodiekj1", "y-1"]
    assert platforms == ["sleeper", "yahoo"]
    # The assistant service resolves through the same helper.
    assert service._linked_identity_keys(42) == (keys, platforms)


def test_linked_identity_keys_fail_closed(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(subscriptions, "get_conn", boom)
    assert subscriptions.linked_identity_keys(42) == ([], [])
    assert service._linked_identity_keys(42) == ([], [])
