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
from dashboard_services.assistant import store
from dashboard_services.assistant.compose import compose_answer, numbers_shown_from
from dashboard_services.assistant.evidence import BUILDERS
from dashboard_services.assistant.model import last_usage
from dashboard_services.assistant.resolve import resolve_player
from dashboard_services.assistant.router import classify

logger = logging.getLogger(__name__)


class CapReached(Exception):
    def __init__(self, caps):
        super().__init__("cap_reached")
        self.caps = caps


def _is_paid(account_key: str) -> bool:
    """Paid tier check.

    Canonical check used: dashboard_services.subscriptions.find_active_personal_plan,
    which queries user_subscriptions for active, unexpired rows keyed by the
    caller's identities (acct:<id> and the bare account id). It fails closed
    (returns None on DB errors), so any failure here defaults to free.
    """
    try:
        from dashboard_services.subscriptions import find_active_personal_plan
        account_id = ctx_mod.resolve_account_id()
        keys = [account_key]
        if account_id is not None:
            keys.append(str(account_id))
        return find_active_personal_plan(keys) is not None
    except Exception:
        return False


def caps_for(account_key: str) -> dict:
    paid = _is_paid(account_key)
    try:
        free_cap = int(os.getenv("ASSISTANT_FREE_MONTHLY_CAP", "25"))
    except Exception:
        free_cap = 25
    try:
        paid_cap = int(os.getenv("ASSISTANT_PAID_MONTHLY_CAP", "250"))
    except Exception:
        paid_cap = 250
    limit = paid_cap if paid else free_cap
    used = store.count_monthly_asked(account_key)
    return {"used": used, "limit": limit, "tier": "paid" if paid else "free"}


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


def ask(account_key: str, message: str, *, platform=None, league_id=None, season=None, thread_id=None, player_id_hint=None) -> dict:
    started = time.time()
    caps = caps_for(account_key)
    if caps["used"] >= caps["limit"]:
        raise CapReached(caps)
    bundle = ctx_mod.resolve_league(account_key, platform=platform, league_id=league_id, season=season)
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
                  "followups": [], "act_link": None, "actions": [], "preference": pref}
        return _finish(answer, log_intent="remember")

    # out of scope
    if intent == "out_of_scope":
        answer = {"verdict": None,
                  "text": "That is outside what I can answer from your league data. I can help with start/sit, trades, buy/sell/hold, player questions, metrics, waivers, or what to do this week.",
                  "receipts": [], "caveat": None,
                  "followups": ["What should I do with my team this week?"], "act_link": None, "actions": []}
        return _finish(answer, log_intent="out_of_scope")

    # entity resolution (clarify short-circuit)
    resolved_entities, ambiguous = _resolve_all(entities, bundle, player_id_hint=player_id_hint)
    if ambiguous is not None:
        answer = {"verdict": None, "text": "Which player do you mean?", "receipts": [], "caveat": None,
                  "followups": [], "act_link": None, "actions": [], "candidates": ambiguous.get("candidates") or []}
        return _finish(answer, log_intent="clarify", entities_out=resolved_entities)

    builder = BUILDERS.get(intent)
    if builder is None:
        answer = {"verdict": None, "text": "I cannot answer that type of question yet.", "receipts": [],
                  "caveat": None, "followups": [], "act_link": None, "actions": []}
        return _finish(answer, log_intent="out_of_scope", entities_out=resolved_entities)

    player_ids = list(resolved_entities.get("player_ids") or []) + list(resolved_entities.get("give_player_ids") or []) + list(resolved_entities.get("receive_player_ids") or [])
    if player_id_hint and not player_ids:
        player_ids = [str(player_id_hint)]
    pack, tools = builder(bundle, resolved_entities, player_ids)

    preferences = store.list_preferences(account_key)

    # Answer caching: user-independent factual answers only.
    answer = None
    cacheable = intent == "metric" or (intent == "player" and not bundle.get("roster"))
    cache_key = None
    if cacheable:
        cache_key = build_ai_cache_key("assistant", {"intent": intent, "entities": resolved_entities, "week": bundle.get("week")}, PROMPT_VERSION)
        try:
            cached = load_cached_ai_text(cache_key)
            if cached:
                answer = json.loads(cached)
        except Exception:
            answer = None
    if answer is None:
        try:
            answer = compose_answer(intent, message, pack, preferences=preferences, assumed_context=assumed, history=history)
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


def add_preference(account_key, text, league_id=None):
    return store.add_preference(account_key, text, league_id=league_id)


def delete_preference(account_key, pref_id):
    return store.delete_preference(account_key, pref_id)


def delete_all_preferences(account_key):
    return store.delete_all_preferences(account_key)
