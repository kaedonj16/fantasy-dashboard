"""Assistant service: orchestration for ask() and the context endpoint."""
from __future__ import annotations

import json
import logging
import os
import time

from dashboard_services.ai.client import AIUnavailableError, AIRateLimitError
from dashboard_services.ai.cache import build_ai_cache_key, load_cached_ai_text, save_cached_ai_text
from dashboard_services.assistant import PROMPT_VERSION
from dashboard_services.assistant import context as ctx_mod
from dashboard_services.assistant import infer, store
from dashboard_services.assistant.compose import compose_answer, followup_pair, numbers_shown_from
from dashboard_services.assistant.evidence import BUILDERS
from dashboard_services.assistant.model import last_usage
from dashboard_services.assistant.resolve import resolve_player
from dashboard_services.assistant.router import classify

logger = logging.getLogger(__name__)


class CapReached(Exception):
    def __init__(self, caps):
        super().__init__("cap_reached")
        self.caps = caps


def _linked_identity_keys(account_id: int):
    """(keys, platforms) a subscription row may be stored under for an account.

    Delegates to subscriptions.linked_identity_keys: the platform identities
    linked to the account in account_identities (Sleeper user id, handle,
    ...). Checkout stamps grants with the platform identity first
    (routes/billing_bp.py's checkout user id is the session's viewer id,
    then the username, and only falls back to acct:<id>), and the admin
    grant script's default form keys rows by Sleeper id/username too. The
    canonical gate (subscriptions.has_premium_access) reaches those rows by
    joining user_subscriptions through this table; a lookup keyed only by
    acct:<id> never sees them. Fails closed: any error yields no keys.
    """
    try:
        from dashboard_services.subscriptions import linked_identity_keys
        return linked_identity_keys(account_id)
    except Exception:
        return [], []


def _plan_lookup_keys(account_key: str):
    """Every identity key this caller's plan may be stored under, plus the
    platforms those identities span.

    The full set the rest of the site resolves through: the account keys
    (acct:<id> and the bare id), the session's viewer identity (checkout
    prefers it when present), and every platform identity linked to the
    account. Order does not matter: find_active_personal_plan matches
    user_id = ANY(keys) and ranks the rows it finds.
    """
    keys: list = []

    def _add(value):
        value = str(value or "").strip()
        if value and value not in keys:
            keys.append(value)

    _add(account_key)
    account_id = ctx_mod.resolve_account_id()
    if account_id is None:
        account_id = _account_id_from_key(account_key)
    if account_id is not None:
        _add(f"acct:{account_id}")
        _add(str(account_id))
    try:
        from flask import has_request_context, session
        if has_request_context():
            _add(session.get("viewer_user_id"))
            _add(session.get("viewer_username"))
    except Exception:
        pass
    platforms = ["sleeper"]
    if account_id is not None:
        linked_keys, linked_platforms = _linked_identity_keys(account_id)
        for key in linked_keys:
            _add(key)
        for platform in linked_platforms:
            if platform not in platforms:
                platforms.append(platform)
    return keys, platforms


def _tier_rank(plan) -> int:
    """Rank of a plan row's assistant tier (higher wins across platforms)."""
    plan_key = str((plan or {}).get("plan_key") or "").strip().lower()
    tier = _TIER_BY_PLAN.get(plan_key, "starter") if plan_key else "free"
    return TIER_ORDER.index(tier)


def _active_plan(account_key: str):
    """The caller's active personal plan row, or None.

    Resolves through the same source of truth as the rest of the site:
    dashboard_services.subscriptions.find_active_personal_plan over the
    caller's full identity key set (see _plan_lookup_keys), across every
    platform the account holds identities on, keeping the best tier found.
    Two account-scoped representations that lookup cannot see are folded in
    after it, mirroring the canonical gate (subscriptions.has_premium_access):
    grandfathered single_league rows live in user_league_subscriptions, not
    user_subscriptions (fold-in decided 2026-10-14: starter), and an active
    PRO free trial grants full PRO while it runs. It fails closed (returns
    None on DB errors), so any failure here defaults to free.
    """
    try:
        from dashboard_services.subscriptions import (
            find_active_personal_plan,
            has_any_user_league_subscription,
            trial_active_for_keys,
        )
        keys, platforms = _plan_lookup_keys(account_key)
        best = None
        for platform in platforms:
            plan = find_active_personal_plan(keys, platform)
            if plan is not None and (best is None or _tier_rank(plan) > _tier_rank(best)):
                best = plan
        if best is not None:
            return best
        account_id = ctx_mod.resolve_account_id()
        if account_id is None:
            account_id = _account_id_from_key(account_key)
        if has_any_user_league_subscription(
            None, account_id=account_id, user_keys=keys,
        ):
            return {"plan_key": "single_league"}
        if trial_active_for_keys(keys):
            return {"plan_key": "trial"}
        return None
    except Exception:
        return None


def _is_paid(account_key: str) -> bool:
    """Paid tier check (any active personal plan)."""
    return _active_plan(account_key) is not None


# Assistant monthly chat caps by pass tier. The purchasable catalog lives in
# routes/billing_bp.py (_STRIPE_PRICES: starter / all_pro / hall_of_fame).
# Grandfathered plan keys fold into the tier at their level: 'single_league'
# was the old entry tier (starter), 'league' sits at all_pro, and 'user' /
# 'combo' are unlimited personal PRO like hall_of_fame. Any other paid plan
# key lands on starter (ASSISTANT_PAID_MONTHLY_CAP still works as the
# starter override). hall_of_fame is positioned as unlimited in the
# catalog, so it gets a high but finite cap (1000) rather than no cap:
# usage still counts and the meter still renders.
TIER_ORDER = ("free", "starter", "all_pro", "hall_of_fame")
TIER_LABELS = {
    "free": "Free",
    "starter": "Starter",
    "all_pro": "All-Pro",
    "hall_of_fame": "Hall of Fame",
}
_TIER_BY_PLAN = {
    "starter": "starter", "single_league": "starter",
    "all_pro": "all_pro", "league": "all_pro",
    "hall_of_fame": "hall_of_fame", "user": "hall_of_fame", "combo": "hall_of_fame",
    # An active PRO free trial is full PRO in the canonical gate (the same
    # scope as the grandfathered 'user' plan), so it tiers as hall_of_fame
    # while it runs.
    "trial": "hall_of_fame",
}
_TIER_CAP_DEFAULTS = {"free": 25, "starter": 100, "all_pro": 250, "hall_of_fame": 1000}
_TIER_CAP_ENV = {
    "free": ("ASSISTANT_FREE_MONTHLY_CAP",),
    "starter": ("ASSISTANT_STARTER_MONTHLY_CAP", "ASSISTANT_PAID_MONTHLY_CAP"),
    "all_pro": ("ASSISTANT_ALL_PRO_MONTHLY_CAP",),
    "hall_of_fame": ("ASSISTANT_HALL_OF_FAME_MONTHLY_CAP",),
}


def _tier_cap(tier: str) -> int:
    for env_name in _TIER_CAP_ENV[tier]:
        raw = os.getenv(env_name)
        if raw is None:
            continue
        try:
            return int(raw)
        except Exception:
            continue
    return _TIER_CAP_DEFAULTS[tier]


def _is_exempt() -> bool:
    """Owner exemption: an admin session is never capped.

    Uses the codebase's existing admin check (dashboard_services.admin_auth):
    a browser session marked admin by the ADMIN_KEY flow or the /admin/login
    password, the same signal analytics uses to self-exclude the operator.
    Usage still counts toward the meter; only enforcement is skipped.
    Outside a request context (background work) nobody is exempt.
    See _is_account_exempt for the account-identity list, which applies in
    any browser, admin session or not.
    """
    try:
        from flask import has_request_context
        if not has_request_context():
            return False
        from dashboard_services.admin_auth import is_admin
        return bool(is_admin())
    except Exception:
        return False


def _exempt_account_ids() -> frozenset:
    """Account ids exempt from assistant caps (ASSISTANT_EXEMPT_ACCOUNT_KEYS).

    Comma-separated account keys in the identity forms the assistant already
    uses: "acct:<id>" entries and bare account ids are both accepted and
    normalized to the bare id (the same normalization the subscription
    lookup applies to caller identities). Parsing mirrors the
    ANALYTICS_EXCLUDE_ACCOUNT_IDS pattern in dashboard_services/analytics.py:
    tolerant of blanks and bad values, and a malformed entry exempts nobody.
    Read per call, like the tier caps above.
    """
    raw = os.getenv("ASSISTANT_EXEMPT_ACCOUNT_KEYS") or ""
    ids = set()
    for part in raw.split(","):
        part = part.strip()
        if part.lower().startswith("acct:"):
            part = part[5:].strip()
        if part.isdigit():
            ids.add(int(part))
    return frozenset(ids)


def _exempt_emails() -> frozenset:
    """Emails exempt from assistant caps (ASSISTANT_EXEMPT_EMAILS), lowercased."""
    raw = os.getenv("ASSISTANT_EXEMPT_EMAILS") or ""
    return frozenset(p.strip().lower() for p in raw.split(",") if p.strip())


def _account_id_from_key(account_key):
    """The bare account id behind an "acct:<id>" key (or a bare id), or None."""
    if not account_key:
        return None
    text = str(account_key).strip()
    if text.lower().startswith("acct:"):
        text = text[5:].strip()
    try:
        return int(text)
    except Exception:
        return None


def _account_email(account_id: int):
    """The account's stored email (accounts.email, set at Google sign-in)."""
    try:
        with store.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT email FROM accounts WHERE id = %s", (account_id,))
                row = cur.fetchone()
        if not row:
            return None
        email = row.get("email") if isinstance(row, dict) else row[0]
        return (str(email).strip().lower() or None) if email else None
    except Exception:
        return None


def _is_account_exempt(account_key: str) -> bool:
    """Account-identity exemption: the account itself is never capped.

    Complements the admin-session check (_is_exempt): an account whose id is
    listed in ASSISTANT_EXEMPT_ACCOUNT_KEYS, or whose stored account email is
    listed in ASSISTANT_EXEMPT_EMAILS, is exempt in any browser, including a
    normal Google sign-in with no admin session (the owner's phone). Usage
    still counts toward the meter; only enforcement is skipped. Fail-closed:
    any lookup error means not exempt.
    """
    try:
        account_id = _account_id_from_key(account_key)
        if account_id is None:
            return False
        if account_id in _exempt_account_ids():
            return True
        emails = _exempt_emails()
        if emails:
            return _account_email(account_id) in emails
        return False
    except Exception:
        return False


def caps_for(account_key: str) -> dict:
    plan = _active_plan(account_key)
    plan_key = None
    if plan is not None:
        plan_key = str(plan.get("plan_key") or "").strip().lower() or None
    if plan_key is None:
        tier = "free"
    else:
        tier = _TIER_BY_PLAN.get(plan_key, "starter")
    tiers = [{"key": key, "label": TIER_LABELS[key], "limit": _tier_cap(key),
              "current": key == tier} for key in TIER_ORDER]
    return {"used": store.count_monthly_asked(account_key),
            "limit": _tier_cap(tier), "tier": tier, "tier_label": TIER_LABELS[tier],
            "plan_key": plan_key, "exempt": _is_exempt() or _is_account_exempt(account_key),
            "tiers": tiers}


def _history_for(thread: dict) -> tuple[list, dict]:
    history, last_entities = [], {}
    try:
        last = store.get_last_log_for_thread(thread.get("thread_id"))
        if last:
            ent = last.get("entities")
            if isinstance(ent, str):
                try:
                    ent = json.loads(ent)
                except Exception:
                    ent = {}
            last_entities = ent or {}
            history = [{"intent": last.get("intent"), "answer": last.get("answer_text")}]
    except Exception:
        pass
    if thread.get("last_entities"):
        last_entities = thread["last_entities"]
    return history, last_entities


def _resolve_all(entities: dict, bundle: dict, player_id_hint=None):
    """Resolve every named player; returns (ids_by_name, ambiguous_result)."""
    roster_ids = bundle.get("roster_player_ids") or set()
    ids: dict[str, str] = {}
    groups = {
        "player_names": list(entities.get("player_names") or []),
        "give_player_names": list(entities.get("give_player_names") or []),
        "receive_player_names": list(entities.get("receive_player_names") or []),
    }
    all_names = groups["player_names"] + groups["give_player_names"] + groups["receive_player_names"]
    for i, name in enumerate(all_names):
        if player_id_hint and i == 0:
            ids[name] = str(player_id_hint)
            continue
        r = resolve_player(name, roster_player_ids=roster_ids)
        if r["status"] == "ambiguous":
            return ids, r
        if r["status"] == "resolved":
            ids[name] = r["player_id"]
    resolved_entities = dict(entities)
    resolved_entities["player_ids"] = [ids[n] for n in groups["player_names"] if n in ids]
    resolved_entities["give_player_ids"] = [ids[n] for n in groups["give_player_names"] if n in ids]
    resolved_entities["receive_player_ids"] = [ids[n] for n in groups["receive_player_names"] if n in ids]
    return resolved_entities, None


def _log_common(bundle, account_key):
    return {
        "account_key": account_key,
        "league_id": bundle.get("league_id"), "platform": bundle.get("platform"),
        "season": bundle.get("season"), "week": bundle.get("week"),
    }


def _open_turn(account_key, message, bundle, entities, last_entities,
               history, assumed, preferences=None):
    """Run one open answering turn (plan, fetch tools, compose from the
    evidence). Returns (answer, pack, tools_called, names): names are the
    players the evidence resolved, which the caller folds into the
    thread entities so the next turn's follow-ups inherit them."""
    from dashboard_services.assistant import open_layer
    if preferences is None:
        preferences = store.list_preferences(account_key)
    context_names: list[str] = []
    for src in (entities.get("player_names"),
                (last_entities or {}).get("player_names")):
        for n in (src or []):
            if n and str(n) not in context_names:
                context_names.append(str(n))
    thread = {
        "context_players": context_names,
        "last_intent": (last_entities or {}).get("_intent"),
        "last_answer": ((history or [{}])[0].get("answer") if history else None),
    }
    answer, pack, tools_called, names = open_layer.answer_open(
        message, bundle, thread=thread, history=history,
        preferences=preferences, assumed_context=assumed)
    return answer, pack, tools_called, names


def ask(account_key: str, message: str, *, platform=None, league_id=None, season=None, thread_id=None, player_id_hint=None) -> dict:
    started = time.time()
    caps = caps_for(account_key)
    if not caps["exempt"] and caps["used"] >= caps["limit"]:
        raise CapReached(caps)
    bundle = ctx_mod.resolve_league(account_key, platform=platform, league_id=league_id, season=season)
    # Builders with a paywall mirror (the breakout board preview) read the
    # caller's paid status from the bundle; caps_for already resolved it
    # (any tier above free is a paid plan).
    bundle["is_paid"] = caps.get("tier") not in (None, "free")
    thread = store.get_or_create_thread(account_key, thread_id, platform=bundle.get("platform"), league_id=bundle.get("league_id"), season=bundle.get("season"))
    tid = thread["thread_id"]
    turn_index = int(thread.get("turn_count") or 0) + 1
    history, last_entities = _history_for(thread)
    assumed = {
        "league_id": bundle.get("league_id"), "league_name": bundle.get("league_name"),
        "platform": bundle.get("platform"), "season": bundle.get("season"),
        "assumed_league_uncertain": bool(bundle.get("assumed_league_uncertain")),
    }
    routed = classify(message, {"league_name": bundle.get("league_name"), "team_name": bundle.get("team_name"), "week": bundle.get("week"), "season": bundle.get("season")}, history)
    intent = routed["intent"]
    entities = dict(routed.get("entities") or {})
    # Follow-ups: short/elliptical messages inherit the thread's last entities.
    if not entities.get("player_names") and last_entities.get("player_names") and len((message or "").split()) <= 6:
        entities["player_names"] = last_entities.get("player_names") or []
        if intent in ("out_of_scope",):
            intent = last_entities.get("_intent") or intent
    # Board follow-ups ("WRs only", "last 4 weeks"): a short message after a
    # leaderboard answer continues the board, inheriting the metric(s) it
    # does not restate; a fresh board question carries its own metric.
    if last_entities.get("_intent") == "leaderboard" and len((message or "").split()) <= 6:
        if intent == "out_of_scope":
            intent = "leaderboard"
        if intent == "leaderboard":
            for _f in ("metric_hint", "second_metric_hint", "board_kind", "family"):
                if not entities.get(_f) and last_entities.get(_f):
                    entities[_f] = last_entities[_f]

    def _finish(answer, *, log_intent, pack=None, tools=None, verdict=None, entities_out=None):
        usage = last_usage()
        log_id = store.write_log(
            **_log_common(bundle, account_key),
            thread_id=tid, turn_index=turn_index, question_text=message,
            intent=log_intent, entities=entities_out or entities, assumed_context=assumed,
            channel="site", proactive=False, trigger=None, input_mode="typed",
            tools_called=tools or [], evidence_pack=pack or {},
            evidence_as_of=(pack or {}).get("as_of") or {},
            answer_text=(answer or {}).get("text"), verdict=verdict if verdict is not None else {"intent": log_intent, "call": (answer or {}).get("verdict")},
            numbers_shown=numbers_shown_from((answer or {}).get("receipts")),
            model=usage.get("model"), router_model=routed.get("router_model"),
            prompt_version=PROMPT_VERSION,
            input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
            cached_tokens=usage.get("cached_tokens"), est_cost_usd=usage.get("est_cost_usd"),
            latency_ms=int((time.time() - started) * 1000),
        )
        store.update_thread(tid, last_entities={**entities, "_intent": log_intent}, turn_count=turn_index)
        return {"thread_id": tid, "log_id": log_id, "intent": log_intent, "answer": answer,
                "caps": {**caps, "used": caps["used"] + 1}}

    # remember
    if intent == "remember":
        text = (routed.get("preference_text") or message or "").strip()
        # Strip a leading "remember ..." phrasing is left to the panel verbatim rule:
        # store the preference text the router extracted, else the raw message.
        pref = store.add_preference(account_key, text, league_id=bundle.get("league_id"))
        answer = {"verdict": None, "text": f"Remembered: {text}", "receipts": [], "caveat": None,
                  "followups": [], "act_link": None, "actions": [], "mentions": [], "preference": pref}
        return _finish(answer, log_intent="remember")

    # out of scope: before declining, probe for site-data anchors (a
    # resolved player, a league team, a stat phrase, a catalog metric).
    # Anchored questions the fixed router did not claim go to the open
    # path; only anchor-less questions keep the canned decline, so plain
    # trivia never spends a model call.
    if intent == "out_of_scope":
        from dashboard_services.assistant import open_layer
        if open_layer.probe(message, bundle, last_entities).get("anchors"):
            answer, pack, otools, names = _open_turn(
                account_key, message, bundle, entities, last_entities,
                history, assumed)
            if names:
                entities["player_names"] = names
            return _finish(answer, log_intent="open", pack=pack, tools=otools,
                           entities_out=dict(entities))
        answer = {"verdict": None,
                  "text": "That is outside what I can answer from your league data. I can help with start/sit, trades, buy/sell/hold, player questions, metrics, waivers, or what to do this week.",
                  "receipts": [], "caveat": None,
                  "followups": [followup_pair("What should I do with my team this week?")], "act_link": None, "actions": [], "mentions": []}
        return _finish(answer, log_intent="out_of_scope")

    # entity resolution (clarify short-circuit)
    resolved_entities, ambiguous = _resolve_all(entities, bundle, player_id_hint=player_id_hint)
    if ambiguous is not None:
        answer = {"verdict": None, "text": "Which player do you mean?", "receipts": [], "caveat": None,
                  "followups": [], "act_link": None, "actions": [], "mentions": [], "candidates": ambiguous.get("candidates") or []}
        return _finish(answer, log_intent="clarify", entities_out=resolved_entities)

    # open: the router committed to the open path (no fixed intent owns
    # the question). Plan and fetch tools, then compose from the evidence.
    if intent == "open":
        answer, pack, otools, names = _open_turn(
            account_key, message, bundle, resolved_entities, last_entities,
            history, assumed)
        if names:
            resolved_entities["player_names"] = names
            entities["player_names"] = names
        return _finish(answer, log_intent="open", pack=pack, tools=otools,
                       entities_out=resolved_entities)

    builder = BUILDERS.get(intent)
    if builder is None:
        answer = {"verdict": None, "text": "I cannot answer that type of question yet.", "receipts": [],
                  "caveat": None, "followups": [], "act_link": None, "actions": [], "mentions": []}
        return _finish(answer, log_intent="out_of_scope", entities_out=resolved_entities)

    player_ids = list(resolved_entities.get("player_ids") or []) + list(resolved_entities.get("give_player_ids") or []) + list(resolved_entities.get("receive_player_ids") or [])
    if player_id_hint and not player_ids:
        player_ids = [str(player_id_hint)]
    pack, tools = builder(bundle, resolved_entities, player_ids)

    preferences = store.list_preferences(account_key)

    # Pack dead end: the fixed metric/leaderboard builders could not
    # place the phrasing at all (the generic "that metric is not
    # tracked" decline: a stat the catalog does not key under that name).
    # Give the open path one shot with full tool access before the
    # no-data answer ships. Curated declines keep their pinned shape:
    # KNOWN_MISSING stats (route rate and friends) name their neighbors
    # on purpose, gated packs keep their gated answer, and data-absent
    # declines have no better source for the open path to fetch.
    if intent in ("metric", "leaderboard") and isinstance(pack, dict) \
            and pack.get("available") is False and not pack.get("gated") \
            and pack.get("missing") in (None, "that metric is not tracked"):
        answer, opack, otools, names = _open_turn(
            account_key, message, bundle, resolved_entities, last_entities,
            history, assumed, preferences=preferences)
        opack_evidence = (opack or {}).get("evidence") or []
        if any(e.get("ok", True) and not e.get("gated") for e in opack_evidence):
            if names:
                resolved_entities["player_names"] = names
                entities["player_names"] = names
            return _finish(answer, log_intent="open", pack=opack, tools=otools,
                           entities_out=resolved_entities)

    # Answer caching: user-independent factual answers only. The key must
    # still scope everything the composed answer actually depends on
    # (Class A audit): the act link is built from the pack's league, the
    # metric receipts order by the viewer's own roster (on_my_roster), and
    # a player answer's recent-form points use the league's scoring
    # settings. An intent/entities/week-only key served one league's (and
    # one manager's) answer to another.
    answer = None
    cacheable = intent == "metric" or (intent == "player" and not bundle.get("roster"))
    cache_key = None
    if cacheable:
        cache_key = build_ai_cache_key("assistant", {
            "intent": intent, "entities": resolved_entities, "week": bundle.get("week"),
            "platform": bundle.get("platform"), "league_id": bundle.get("league_id"),
            "season": bundle.get("season"),
            "roster": sorted(str(p) for p in (bundle.get("roster_player_ids") or set())),
        }, PROMPT_VERSION)
        try:
            cached = load_cached_ai_text(cache_key)
            if cached:
                answer = json.loads(cached)
        except Exception:
            answer = None
    if answer is None:
        try:
            # The confirmed risk profile rides along only when one exists,
            # so profile-less asks compose exactly as before.
            profile = infer.profile_from_preferences(preferences)
            answer = compose_answer(intent, message, pack, preferences=preferences,
                                    assumed_context=assumed, history=history,
                                    **({"risk_profile": profile} if profile else {}))
        except AIRateLimitError as exc:
            raise AIUnavailableError(str(exc)) from exc
        if cache_key and answer:
            try:
                save_cached_ai_text(cache_key, json.dumps(answer))
            except Exception:
                pass
    return _finish(answer, log_intent=intent, pack=pack, tools=tools, entities_out=resolved_entities)


def _parse_json_col(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return None
    return value


def get_thread(account_key: str, thread_id: str) -> dict | None:
    """One of the account's threads plus its chat history, for restoring
    the conversation after a page change. None when the thread does not
    exist or belongs to another account."""
    if not thread_id:
        return None
    row = store.get_thread_row(str(thread_id))
    if not row or row.get("account_key") != account_key:
        return None
    return {"thread_id": row["thread_id"],
            "league": {"platform": row.get("platform"), "league_id": row.get("league_id"),
                       "season": row.get("season")},
            "messages": store.list_thread_messages(account_key, str(thread_id))}


def get_context(account_key: str, *, platform=None, league_id=None, season=None) -> dict:
    bundle = ctx_mod.resolve_league(account_key, platform=platform, league_id=league_id, season=season)
    caps = caps_for(account_key)
    league_out = None
    if bundle.get("league_id"):
        league_out = {"platform": bundle.get("platform"), "league_id": bundle.get("league_id"),
                      "season": bundle.get("season"), "league_name": bundle.get("league_name"),
                      "team_name": bundle.get("team_name"), "week": bundle.get("week")}
    teaser = None
    if bundle.get("league_id"):
        existing = store.find_proactive_digest(account_key, bundle.get("league_id"), bundle.get("week"))
        if existing is not None:
            verdict = _parse_json_col(existing.get("verdict")) or {}
            stored_answer = verdict.get("answer") if isinstance(verdict, dict) else None
            if stored_answer:
                actions = stored_answer.get("actions") or []
                lead = (actions[0].get("headline") if actions else None) or stored_answer.get("text")
                teaser = {"log_id": existing.get("log_id"), "kind": "digest",
                          "title": f"Your Week {bundle.get('week')} moves are ready",
                          "lead": lead, "answer": stored_answer}
            # else: marker row (empty digest) - do not recompose all week.
        else:
            builder = BUILDERS["this_week"]
            pack, tools = builder(bundle, {}, [])
            actions = (pack or {}).get("actions") or []
            if actions:
                preferences = store.list_preferences(account_key)
                try:
                    answer = compose_answer("this_week", "What should I do with my team this week?", pack,
                                            preferences=preferences, assumed_context={"league_id": bundle.get("league_id")})
                except Exception:
                    answer = None
                if answer:
                    usage = last_usage()
                    log_id = store.write_log(
                        **_log_common(bundle, account_key),
                        thread_id=None, turn_index=None, question_text=None,
                        intent="this_week", entities={}, assumed_context={"league_id": bundle.get("league_id")},
                        channel="site", proactive=True, trigger="digest", input_mode="typed",
                        tools_called=tools, evidence_pack=pack, evidence_as_of=pack.get("as_of") or {},
                        answer_text=answer.get("text"),
                        verdict={"intent": "this_week", "call": answer.get("verdict"), "answer": answer},
                        numbers_shown=numbers_shown_from(answer.get("receipts")),
                        model=usage.get("model"), router_model=None, prompt_version=PROMPT_VERSION,
                        input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
                        cached_tokens=usage.get("cached_tokens"), est_cost_usd=usage.get("est_cost_usd"),
                        latency_ms=usage.get("latency_ms"),
                    )
                    teaser = {"log_id": log_id, "kind": "digest",
                              "title": f"Your Week {bundle.get('week')} moves are ready",
                              "lead": (actions[0].get("headline") if actions else None) or answer.get("text"),
                              "answer": answer}
            if teaser is None and not actions:
                # Marker row so we do not recompose an empty digest all week.
                store.write_log(
                    **_log_common(bundle, account_key),
                    thread_id=None, turn_index=None, question_text=None,
                    intent="this_week", entities={}, assumed_context={"league_id": bundle.get("league_id")},
                    channel="site", proactive=True, trigger="digest", input_mode="typed",
                    tools_called=tools, evidence_pack=pack, evidence_as_of=pack.get("as_of") or {},
                    answer_text=None, verdict=None, numbers_shown={},
                    model=None, router_model=None, prompt_version=PROMPT_VERSION,
                )
    return {"league": league_out, "caps": caps, "teaser": teaser}


# -- feedback / corrections / preferences (thin wraps over store) -------------

def set_thumb(log_id, account_key, thumb):
    return store.set_thumb(log_id, account_key, thumb)


def set_correction(log_id, account_key, *, category, text=None):
    return store.set_correction(log_id, account_key, category=category, text=text)


def apply_correction(log_id, account_key, *, category, text=None) -> dict | None:
    """Record a correction; for wrong-data claims, re-check the answer.

    Populates corrected_claim (the user's own words when given, else the
    call on record that is being disputed) and reask_flag (wrong-data claims
    only). A wrong-data correction re-runs the answer's evidence builder
    against the current league context and recomposes: when the fresh
    evidence changes the call or a cited number, the revised answer is
    logged as a new row chained to the original via revises_log_id and
    returned as `revision` for the UI to append to the thread. When the
    re-check cannot run (no league context, no builder, model error), the
    result says so (`recheck: "unavailable"`) and the caller must not
    promise one. Returns None when the log row is not the caller's.
    """
    row = store.get_log_full(log_id, account_key)
    if row is None:
        return None
    verdict = row.get("verdict") if isinstance(row.get("verdict"), dict) else {}
    corrected_claim = (str(text or "").strip()
                       or str(verdict.get("call") or "").strip()
                       or str(row.get("answer_text") or "").strip() or None)
    reask = category == "wrong_data"
    store.set_correction(log_id, account_key, category=category, text=text,
                         corrected_claim=corrected_claim, reask_flag=reask)
    result = {"ok": True, "corrected_claim": corrected_claim, "reask_flag": reask,
              "revision": None, "recheck": None}
    if not reask:
        return result
    result["recheck"] = "unavailable"
    intent = row.get("intent") or ""
    builder = BUILDERS.get(intent)
    if builder is None or not row.get("league_id"):
        return result
    try:
        bundle = ctx_mod.resolve_league(account_key, platform=row.get("platform"),
                                        league_id=row.get("league_id"), season=row.get("season"))
        if not bundle.get("ctx"):
            return result
        entities = row.get("entities") if isinstance(row.get("entities"), dict) else {}
        player_ids = (list(entities.get("player_ids") or [])
                      + list(entities.get("give_player_ids") or [])
                      + list(entities.get("receive_player_ids") or []))
        pack, tools = builder(bundle, entities, player_ids)
        if isinstance(pack, dict) and pack.get("available") is False:
            return result  # the evidence no longer reproduces this answer at all
        preferences = store.list_preferences(account_key)
        answer = compose_answer(intent, row.get("question_text") or "", pack,
                                preferences=preferences,
                                assumed_context={"league_id": row.get("league_id")}, history=None)
    except Exception:
        logger.info("[assistant] correction re-check could not run for log %s", log_id, exc_info=True)
        return result
    old_call = verdict.get("call")
    old_numbers = row.get("numbers_shown") if isinstance(row.get("numbers_shown"), dict) else {}
    new_numbers = numbers_shown_from(answer.get("receipts"))
    if answer.get("verdict") == old_call and new_numbers == old_numbers:
        result["recheck"] = "unchanged"
        return result
    usage = last_usage()
    new_log_id = store.write_log(
        account_key=account_key, league_id=row.get("league_id"), platform=row.get("platform"),
        season=row.get("season"), week=row.get("week"),
        thread_id=row.get("thread_id"), turn_index=row.get("turn_index"),
        question_text=row.get("question_text"), intent=intent, entities=entities,
        assumed_context={}, channel="site", proactive=True, trigger="correction_recheck",
        input_mode="typed", tools_called=tools, evidence_pack=pack,
        evidence_as_of=pack.get("as_of") or {}, answer_text=answer.get("text"),
        verdict={"intent": intent, "call": answer.get("verdict")},
        numbers_shown=new_numbers, model=usage.get("model"), router_model=None,
        prompt_version=PROMPT_VERSION, revises_log_id=int(log_id),
    )
    result["recheck"] = "changed"
    result["revision"] = {"log_id": new_log_id, "answer": answer}
    return result


def get_log(log_id, account_key):
    return store.get_log(log_id, account_key)


def list_preferences(account_key):
    return store.list_preferences(account_key)


def preference_panel(account_key) -> dict:
    """Everything the "What BR remembers about you" view renders:
    stored preferences, the confirmed risk profile in plain language,
    and fresh inferred-preference proposals.

    Proposals are computed here, when the panel is opened: a cheap
    deterministic pass over the account's own recent advice log, synced
    into pending proposal rows (confirm/dismiss state persists there).
    Fail-soft by design: a proposal problem never breaks the panel.
    """
    prefs = store.list_preferences(account_key)
    profile = infer.profile_from_preferences(prefs)
    proposals = []
    try:
        rows = store.list_log_for_inference(account_key)
        candidates = infer.candidates_from_log(rows)
        proposals = store.sync_proposals(account_key, candidates)
    except Exception:
        logger.debug("[assistant] proposal computation failed", exc_info=True)
        proposals = []
    return {"preferences": prefs, "proposals": proposals,
            "risk_profile": profile, "profile_lines": infer.profile_lines(profile)}


def confirm_proposal(account_key, proposal_id):
    """Confirm a proposal into a stored preference; returns the fresh
    panel payload (without recomputing proposals), or None."""
    pref = store.confirm_proposal(account_key, proposal_id)
    if pref is None:
        return None
    prefs = store.list_preferences(account_key)
    profile = infer.profile_from_preferences(prefs)
    return {"preference": pref, "preferences": prefs,
            "proposals": store.list_proposals(account_key, "pending"),
            "risk_profile": profile, "profile_lines": infer.profile_lines(profile)}


def dismiss_proposal(account_key, proposal_id) -> bool:
    return store.dismiss_proposal(account_key, proposal_id)


def add_preference(account_key, text, league_id=None):
    return store.add_preference(account_key, text, league_id=league_id)


def delete_preference(account_key, pref_id):
    return store.delete_preference(account_key, pref_id)


def delete_all_preferences(account_key):
    return store.delete_all_preferences(account_key)
