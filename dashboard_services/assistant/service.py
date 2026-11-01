"""Assistant service: orchestration for ask() and the context endpoint."""
from __future__ import annotations

import json
import logging
import os
import threading
import time

from dashboard_services.ai.client import AIUnavailableError, AIRateLimitError
from dashboard_services.ai.cache import build_ai_cache_key, load_cached_ai_text, save_cached_ai_text
from dashboard_services.assistant import PROMPT_VERSION
from dashboard_services.assistant import context as ctx_mod
from dashboard_services.assistant import infer, store
from dashboard_services.assistant.compose import compose_answer, followup_pair, numbers_shown_from
from dashboard_services.assistant.evidence import BUILDERS
from dashboard_services.assistant.model import last_usage
from dashboard_services.assistant.resolve import _normalize, resolve_player
from dashboard_services.assistant.router import classify, partner_want_reply, positions_reply

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


def _plan_state(account_key: str) -> tuple[str, str | None]:
    """(tier, plan_key) from the account's active plan: the subscription
    DB chain behind caps_for, split out so ask() can cache just this
    plan-derived state per conversation (see _resolve_snapshot)."""
    plan = _active_plan(account_key)
    plan_key = None
    if plan is not None:
        plan_key = str(plan.get("plan_key") or "").strip().lower() or None
    if plan_key is None:
        tier = "free"
    else:
        tier = _TIER_BY_PLAN.get(plan_key, "starter")
    return tier, plan_key


def _caps_dict(account_key: str, tier: str, plan_key, used: int) -> dict:
    tiers = [{"key": key, "label": TIER_LABELS[key], "limit": _tier_cap(key),
              "current": key == tier} for key in TIER_ORDER]
    return {"used": used,
            "limit": _tier_cap(tier), "tier": tier, "tier_label": TIER_LABELS[tier],
            "plan_key": plan_key, "exempt": _is_exempt() or _is_account_exempt(account_key),
            "tiers": tiers}


def caps_for(account_key: str) -> dict:
    tier, plan_key = _plan_state(account_key)
    return _caps_dict(account_key, tier, plan_key, store.count_monthly_asked(account_key))


# -- Per-conversation resolve snapshot -------------------------------------
# ask() used to re-pay the full preamble on every message: the plan DB
# chain (_active_plan), resolve_league, and list_preferences. Within one
# conversation those change only when a preference is stored (invalidated
# explicitly below) or a plan changes (bounded by the TTL). The monthly
# "used" count is NEVER cached: cap enforcement re-reads it per message,
# and env-driven caps/exemptions are likewise re-evaluated per message
# via _caps_dict. Thread/history state is never cached either: it changes
# every turn by design.
_RESOLVE_SNAPSHOT_TTL = 90  # seconds; bounds plan/league staleness mid-thread
_RESOLVE_SNAPSHOT_MAX = 128  # entries, oldest evicted first
_resolve_snapshot_cache: dict = {}
_resolve_snapshot_lock = threading.Lock()


def _clear_resolve_snapshot_cache() -> None:
    with _resolve_snapshot_lock:
        _resolve_snapshot_cache.clear()


def _invalidate_resolve_snapshot(account_key: str) -> None:
    with _resolve_snapshot_lock:
        for key in [k for k in _resolve_snapshot_cache if k[0] == account_key]:
            _resolve_snapshot_cache.pop(key, None)


def _bundle_copy(bundle: dict) -> dict:
    """Shallow copy the caller may stamp on (ask() sets is_paid /
    forced_player_ids); the nested ctx/roster stay shared by reference,
    exactly as resolve_league's own callers already share them."""
    out = dict(bundle)
    if isinstance(out.get("roster_player_ids"), set):
        out["roster_player_ids"] = set(out["roster_player_ids"])
    if isinstance(out.get("viewer"), dict):
        out["viewer"] = dict(out["viewer"])
    return out


def _resolve_snapshot(account_key: str, thread_id, platform, league_id, season):
    """(bundle_copy, tier, plan_key, preferences) for one conversation.

    Keyed on the caller-supplied thread id and league params, so a league
    switch mid-thread misses and re-resolves. Compute happens outside the
    lock; a race may compute twice, which is the same tolerance as the
    app.py TTL caches.
    """
    key = (account_key, thread_id or "", platform or "",
           str(league_id or ""), season or "")
    now = time.time()
    with _resolve_snapshot_lock:
        hit = _resolve_snapshot_cache.get(key)
        if hit and now - hit[0] < _RESOLVE_SNAPSHOT_TTL:
            _ts, bundle, tier, plan_key, prefs = hit
            return _bundle_copy(bundle), tier, plan_key, list(prefs)
    bundle = ctx_mod.resolve_league(account_key, platform=platform,
                                    league_id=league_id, season=season)
    tier, plan_key = _plan_state(account_key)
    prefs = store.list_preferences(account_key)
    with _resolve_snapshot_lock:
        if len(_resolve_snapshot_cache) >= _RESOLVE_SNAPSHOT_MAX:
            oldest = min(_resolve_snapshot_cache,
                         key=lambda k: _resolve_snapshot_cache[k][0])
            _resolve_snapshot_cache.pop(oldest, None)
        _resolve_snapshot_cache[key] = (now, bundle, tier, plan_key, prefs)
    return _bundle_copy(bundle), tier, plan_key, list(prefs)


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


def _resolve_all(entities: dict, bundle: dict, player_id_hint=None, forced_ids=None):
    """Resolve every named player.

    Returns (resolved_entities, ambiguous_result, ambiguous_name); the
    first element is the partial ids dict when an ambiguity stops the
    walk. forced_ids maps normalized names to player ids the thread has
    already settled (a clarification pick, remembered on the thread):
    those names resolve to the settled id without asking again.
    """
    roster_ids = bundle.get("roster_player_ids") or set()
    forced = {_normalize(k): str(v) for k, v in (forced_ids or {}).items()}
    ids: dict[str, str] = {}
    groups = {
        "player_names": list(entities.get("player_names") or []),
        "give_player_names": list(entities.get("give_player_names") or []),
        "receive_player_names": list(entities.get("receive_player_names") or []),
    }
    all_names = groups["player_names"] + groups["give_player_names"] + groups["receive_player_names"]
    for i, name in enumerate(all_names):
        settled = forced.get(_normalize(name))
        if settled:
            ids[name] = settled
            continue
        if player_id_hint and i == 0:
            ids[name] = str(player_id_hint)
            continue
        r = resolve_player(name, roster_player_ids=roster_ids)
        if r["status"] == "ambiguous":
            return ids, r, name
        if r["status"] == "resolved":
            ids[name] = r["player_id"]
    resolved_entities = dict(entities)
    resolved_entities["player_ids"] = [ids[n] for n in groups["player_names"] if n in ids]
    resolved_entities["give_player_ids"] = [ids[n] for n in groups["give_player_names"] if n in ids]
    resolved_entities["receive_player_ids"] = [ids[n] for n in groups["receive_player_names"] if n in ids]
    return resolved_entities, None, None


def _pending_from_answer(answer, *, intent, entities, question):
    """The pending-clarification record for an answer that came back
    with candidates (the open path's clarify shape), or None. The next
    turn's pick resumes this request instead of starting over."""
    candidates = (answer or {}).get("candidates") or []
    if not candidates:
        return None
    return {"question": question, "intent": intent, "entities": entities,
            "ambiguous_name": None, "candidates": candidates}


def _pending_pick(pending: dict, message, player_id_hint):
    """The candidate a follow-up turn picked for a pending clarification,
    as (player_id, candidate), or None when the turn is not a pick.

    Picks arrive as a candidate chip tap (the message is the candidate's
    name and the player id rides along as the hint) or as a typed name.
    A typed name only picks when it matches exactly one candidate: when
    the namesakes share the name itself, the bare name is not a pick
    and the clarification stands rather than guessing.
    """
    candidates = [c for c in (pending.get("candidates") or []) if isinstance(c, dict)]
    if not candidates:
        return None
    if player_id_hint:
        for c in candidates:
            if str(c.get("player_id")) == str(player_id_hint):
                return str(player_id_hint), c
    msg_norm = _normalize(message or "")
    if msg_norm:
        named = [c for c in candidates
                 if c.get("name") and _normalize(c["name"]) == msg_norm]
        if len(named) == 1:
            return str(named[0].get("player_id")), named[0]
    return None


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
    bundle, _tier, _plan_key, preferences = _resolve_snapshot(
        account_key, thread_id, platform, league_id, season)
    caps = _caps_dict(account_key, _tier, _plan_key,
                      store.count_monthly_asked(account_key))
    if not caps["exempt"] and caps["used"] >= caps["limit"]:
        raise CapReached(caps)
    # Builders with a paywall mirror (the breakout board preview) read the
    # caller's paid status from the bundle; caps_for already resolved it
    # (any tier above free is a paid plan).
    bundle["is_paid"] = caps.get("tier") not in (None, "free")
    thread = store.get_or_create_thread(account_key, thread_id, platform=bundle.get("platform"), league_id=bundle.get("league_id"), season=bundle.get("season"))
    tid = thread["thread_id"]
    turn_index = int(thread.get("turn_count") or 0) + 1
    history, last_entities = _history_for(thread)
    # Thread name memory: names a clarification already settled on this
    # thread (normalized name -> player id), plus the pending
    # clarification awaiting a pick, if the last turn asked one.
    resolved_names: dict[str, str] = {}
    _stored_names = (last_entities or {}).get("resolved_names")
    if isinstance(_stored_names, dict):
        resolved_names = {_normalize(k): str(v) for k, v in _stored_names.items()
                          if k and v}
    pending_clarify = (last_entities or {}).get("pending_clarify")
    if not isinstance(pending_clarify, dict):
        pending_clarify = None
    # A pending slot clarification: the last turn's answer was itself a
    # question (a pack that dead-ended asking for one missing entity).
    pending_slot = (last_entities or {}).get("pending_slot")
    if not isinstance(pending_slot, dict):
        pending_slot = None
    assumed = {
        "league_id": bundle.get("league_id"), "league_name": bundle.get("league_name"),
        "platform": bundle.get("platform"), "season": bundle.get("season"),
        "assumed_league_uncertain": bool(bundle.get("assumed_league_uncertain")),
    }
    routed = classify(message, {"league_name": bundle.get("league_name"), "team_name": bundle.get("team_name"), "week": bundle.get("week"), "season": bundle.get("season")}, history)
    intent = routed["intent"]
    entities = dict(routed.get("entities") or {})
    # Clarification resume: the last turn asked "which player do you
    # mean?" and this turn is the pick (a candidate chip tap or a typed
    # name). Restore the pending request wholesale: its intent and its
    # entities, with the picked player settled for the ambiguous name,
    # so the original question executes instead of the pick degrading
    # into a bare player mention. The pick is also remembered on the
    # thread, so re-asking the original question never re-clarifies.
    forced_ids = dict(resolved_names)
    resumed = False
    if pending_clarify:
        pick = _pending_pick(pending_clarify, message, player_id_hint)
        if pick is not None:
            picked_id, picked = pick
            intent = pending_clarify.get("intent") or intent
            entities = dict(pending_clarify.get("entities") or {})
            # The turn answers the original question, not the bare pick:
            # compose and the log both key off the message.
            message = pending_clarify.get("question") or message
            amb_name = pending_clarify.get("ambiguous_name") or picked.get("name")
            for nm in (amb_name, picked.get("name")):
                if nm:
                    forced_ids[_normalize(nm)] = picked_id
                    resolved_names[_normalize(nm)] = picked_id
            resumed = True
    if forced_ids:
        bundle["forced_player_ids"] = forced_ids
    # Slot-clarification resume: the last turn's answer asked for the one
    # entity its pack was missing ("Tell me a position and I will rank
    # the defenses against it."). A reply that is only position(s) fills
    # the slot and re-runs the pending request with its focus intact,
    # instead of routing cold (a bare "wr and rb" otherwise degrades
    # into an unrelated board). Any other reply lets the slot lapse:
    # thread state is rebuilt every turn, so an unregistered slot is
    # gone after one turn.
    slot_resumed = False
    if pending_slot and not resumed:
        reply_positions = positions_reply(message)
        if reply_positions:
            intent = pending_slot.get("intent") or intent
            entities = dict(pending_slot.get("entities") or {})
            entities["positions"] = reply_positions
            entities["position"] = reply_positions[0]
            if pending_slot.get("team") and not entities.get("team"):
                entities["team"] = pending_slot["team"]
            # The turn answers the pending question, not the bare reply:
            # compose and the log both key off the message.
            message = pending_slot.get("question") or message
            slot_resumed = True
    # Partner-want resume: right after a trade-target answer, a reply
    # like "he wants a rb" names what the other manager wants back,
    # which is the user's outgoing side. Re-run the pending trade
    # question with that position as the give constraint; the target
    # (and with it the partner, the target's owner) rides in from the
    # last turn's entities, and the pack is marked partner-scoped so a
    # constrained no-fit declines at that partner before any widening.
    partner_resumed = False
    if not resumed and not slot_resumed \
            and (last_entities or {}).get("_intent") == "trade_finding" \
            and (last_entities or {}).get("player_names") \
            and not (last_entities or {}).get("give_player_names"):
        want_pos = partner_want_reply(message)
        if want_pos:
            intent = "trade_finding"
            entities = {"player_names": list(last_entities["player_names"]),
                        "target_position": last_entities.get("target_position"),
                        "give_position": want_pos,
                        "partner_scoped": True}
            partner_resumed = True
    if not resumed and not slot_resumed and not partner_resumed:
        # Defense follow-ups keep the thread's defense in focus: a
        # matchup/defenses board that names no team inherits the team
        # the thread was just discussing. Scoped to the defense board
        # kinds: on any other board a team is a player filter, never a
        # focus, so a plain board question after a defense answer keeps
        # its own (teamless) meaning.
        if intent == "leaderboard" and not entities.get("team") \
                and entities.get("board_kind") in ("matchup", "defenses") \
                and (last_entities or {}).get("team"):
            entities["team"] = last_entities["team"]
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

    def _finish(answer, *, log_intent, pack=None, tools=None, verdict=None, entities_out=None, pending=None, pending_slot=None):
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
        thread_entities = {**entities, "_intent": log_intent,
                           "resolved_names": resolved_names}
        if pending is not None:
            thread_entities["pending_clarify"] = pending
        if pending_slot is not None:
            thread_entities["pending_slot"] = pending_slot
        store.update_thread(tid, last_entities=thread_entities, turn_count=turn_index)
        return {"thread_id": tid, "log_id": log_id, "intent": log_intent, "answer": answer,
                "caps": {**caps, "used": caps["used"] + 1}}

    # remember
    if intent == "remember":
        text = (routed.get("preference_text") or message or "").strip()
        # Strip a leading "remember ..." phrasing is left to the panel verbatim rule:
        # store the preference text the router extracted, else the raw message.
        pref = store.add_preference(account_key, text, league_id=bundle.get("league_id"))
        _invalidate_resolve_snapshot(account_key)
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
                history, assumed, preferences=preferences)
            if names:
                entities["player_names"] = names
            return _finish(answer, log_intent="open", pack=pack, tools=otools,
                           entities_out=dict(entities),
                           pending=_pending_from_answer(
                               answer, intent="open", entities=dict(entities),
                               question=message))
        answer = {"verdict": None,
                  "text": "That is outside what I can answer from your league data. I can help with start/sit, trades, buy/sell/hold, player questions, metrics, waivers, or what to do this week.",
                  "receipts": [], "caveat": None,
                  "followups": [followup_pair("What should I do with my team this week?")], "act_link": None, "actions": [], "mentions": []}
        return _finish(answer, log_intent="out_of_scope")

    # entity resolution (clarify short-circuit). On a resumed turn the
    # forced map carries the pick; the bare hint would otherwise be
    # stamped onto the first name in the list, which may not be the
    # name that was ambiguous.
    resolved_entities, ambiguous, ambiguous_name = _resolve_all(
        entities, bundle,
        player_id_hint=None if resumed else player_id_hint,
        forced_ids=forced_ids)
    if ambiguous is not None:
        answer = {"verdict": None, "text": "Which player do you mean?", "receipts": [], "caveat": None,
                  "followups": [], "act_link": None, "actions": [], "mentions": [], "candidates": ambiguous.get("candidates") or []}
        pending = {"question": message, "intent": intent, "entities": entities,
                   "ambiguous_name": ambiguous_name,
                   "candidates": ambiguous.get("candidates") or []}
        return _finish(answer, log_intent="clarify", entities_out=resolved_entities,
                       pending=pending)

    # open: the router committed to the open path (no fixed intent owns
    # the question). Plan and fetch tools, then compose from the evidence.
    if intent == "open":
        answer, pack, otools, names = _open_turn(
            account_key, message, bundle, resolved_entities, last_entities,
            history, assumed, preferences=preferences)
        if names:
            resolved_entities["player_names"] = names
            entities["player_names"] = names
        return _finish(answer, log_intent="open", pack=pack, tools=otools,
                       entities_out=resolved_entities,
                       pending=_pending_from_answer(
                           answer, intent="open", entities=resolved_entities,
                           question=message))

    builder = BUILDERS.get(intent)
    if builder is None:
        answer = {"verdict": None, "text": "I cannot answer that type of question yet.", "receipts": [],
                  "caveat": None, "followups": [], "act_link": None, "actions": [], "mentions": []}
        return _finish(answer, log_intent="out_of_scope", entities_out=resolved_entities)

    player_ids = list(resolved_entities.get("player_ids") or []) + list(resolved_entities.get("give_player_ids") or []) + list(resolved_entities.get("receive_player_ids") or [])
    if player_id_hint and not player_ids:
        player_ids = [str(player_id_hint)]
    pack, tools = builder(bundle, resolved_entities, player_ids)

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
                           entities_out=resolved_entities,
                           pending=_pending_from_answer(
                               answer, intent="open", entities=resolved_entities,
                               question=message))

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
    # A pack that dead-ends asking for its missing slot registers a
    # pending slot, so the user's bare position reply resumes this exact
    # request (focus team included) instead of routing cold. Only the
    # matchup board's position ask exists today; the record is generic
    # (pending intent + entities + focus) so the next slot kind reuses it.
    pending_slot_out = None
    if isinstance(pack, dict) and pack.get("available") is False \
            and pack.get("missing") == "a position to rank defenses against":
        pending_slot_out = {
            "kind": "matchup_position",
            "question": message,
            "intent": "leaderboard",
            "entities": {**entities, "board_kind": "matchup"},
            "team": entities.get("team"),
        }
    return _finish(answer, log_intent=intent, pack=pack, tools=tools,
                   entities_out=resolved_entities,
                   pending_slot=pending_slot_out)


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


# -- Proactive digest: composed in the background, never inside the GET ---
# get_context used to run the this_week builder + an LLM compose inline
# whenever no digest row existed yet, so an ordinary page load could pay
# a model call. Now the GET only reads: an existing digest row serves the
# teaser, and a missing one kicks ONE bounded background build (per-key
# single-flight, semaphore-capped like _PLAYOFF_SIM_SEM in app.py) and
# returns teaser=None; the teaser appears on a later load. The once-per-
# week marker is unchanged: _build_digest writes the same real-or-marker
# proactive log row the old inline path wrote.
_DIGEST_INFLIGHT: set = set()
_DIGEST_LOCK = threading.Lock()
_DIGEST_SEM = threading.BoundedSemaphore(2)  # max concurrent digest builds per worker


def _spawn_digest_thread(fn) -> None:
    threading.Thread(target=fn, daemon=True).start()


def _build_digest(account_key: str, bundle: dict) -> None:
    """The old inline get_context compose, verbatim: build the this_week
    pack, compose when it has actions, store the digest row; store the
    empty marker row when it has none. Runs on a background thread with
    the bundle resolved on the request thread (no session access here)."""
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
            store.write_log(
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
            return
    if not actions:
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


def _kick_digest(account_key: str, bundle: dict) -> None:
    """Single-flight background digest build. A dropped kick (semaphore
    busy) clears its mark so the next context load can kick again."""
    key = (account_key, str(bundle.get("league_id") or ""), bundle.get("week"))
    with _DIGEST_LOCK:
        if key in _DIGEST_INFLIGHT:
            return
        _DIGEST_INFLIGHT.add(key)
    if not _DIGEST_SEM.acquire(blocking=False):
        with _DIGEST_LOCK:
            _DIGEST_INFLIGHT.discard(key)
        return

    # Aggregate heavy cap (app.py's _BG_HEAVY_SEM, resolved lazily like
    # the other app internals this package uses at request time): a
    # digest build is one of the heavy background kicks the per-worker
    # ceiling covers, on top of this mechanism's own cap.
    from app import (
        _heavy_job_end,
        _heavy_job_start,
        _heavy_kick_acquire,
        _heavy_kick_release,
    )

    agg_token = _heavy_kick_acquire()
    if agg_token is None:
        _DIGEST_SEM.release()
        with _DIGEST_LOCK:
            _DIGEST_INFLIGHT.discard(key)
        return

    def _run():
        prev = _heavy_job_start(agg_token)
        try:
            _build_digest(account_key, bundle)
        except Exception:
            logger.debug("[assistant] background digest build failed", exc_info=True)
        finally:
            _DIGEST_SEM.release()
            _heavy_job_end(agg_token, prev)
            with _DIGEST_LOCK:
                _DIGEST_INFLIGHT.discard(key)

    try:
        _spawn_digest_thread(_run)
    except Exception:
        _DIGEST_SEM.release()
        _heavy_kick_release(agg_token)
        with _DIGEST_LOCK:
            _DIGEST_INFLIGHT.discard(key)


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
            # No digest row yet: kick the bounded background build and
            # serve without a teaser. Composing here would put an LLM
            # call inside this GET on an ordinary page load.
            _kick_digest(account_key, dict(bundle))
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
    _invalidate_resolve_snapshot(account_key)
    prefs = store.list_preferences(account_key)
    profile = infer.profile_from_preferences(prefs)
    return {"preference": pref, "preferences": prefs,
            "proposals": store.list_proposals(account_key, "pending"),
            "risk_profile": profile, "profile_lines": infer.profile_lines(profile)}


def dismiss_proposal(account_key, proposal_id) -> bool:
    return store.dismiss_proposal(account_key, proposal_id)


def add_preference(account_key, text, league_id=None):
    pref = store.add_preference(account_key, text, league_id=league_id)
    _invalidate_resolve_snapshot(account_key)
    return pref


def delete_preference(account_key, pref_id):
    ok = store.delete_preference(account_key, pref_id)
    _invalidate_resolve_snapshot(account_key)
    return ok


def delete_all_preferences(account_key):
    ok = store.delete_all_preferences(account_key)
    _invalidate_resolve_snapshot(account_key)
    return ok
