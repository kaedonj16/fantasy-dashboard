"""Intent router: cheap model classification with a keyword fallback."""
from __future__ import annotations

import json
import logging
import re

from dashboard_services.ai.client import AIUnavailableError, AIRateLimitError
from dashboard_services.assistant import ROUTER_PROMPT_VERSION
from dashboard_services.assistant.model import call_model_json

logger = logging.getLogger(__name__)

INTENTS = ["start_sit", "trade_eval", "buy_sell_hold", "player", "metric", "waivers", "this_week", "remember", "out_of_scope"]

SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENTS},
        "player_names": {"type": "array", "items": {"type": "string"}},
        "give_player_names": {"type": "array", "items": {"type": "string"}},
        "receive_player_names": {"type": "array", "items": {"type": "string"}},
        "picks": {"type": "array", "items": {"type": "string"}},
        "other_team_hint": {"type": ["string", "null"]},
        "metric_hint": {"type": ["string", "null"]},
        "week": {"type": ["integer", "null"]},
        "preference_text": {"type": ["string", "null"]},
    },
    "required": ["intent", "player_names", "give_player_names", "receive_player_names", "picks", "other_team_hint", "metric_hint", "week", "preference_text"],
    "additionalProperties": False,
}

SYSTEM = (
    "You route fantasy football questions for the BR Fantasy assistant. "
    "Classify into exactly one intent and extract entities verbatim from the message. "
    "start_sit: who to start. trade_eval: a proposed trade with give and receive sides. "
    "buy_sell_hold: buy/sell/hold on a player. player: a general player question. "
    "metric: a metric definition or leaderboard question (metric_hint is the metric name). "
    "waivers: adds, drops, or FAAB. this_week: what to do with my team this week. "
    "remember: the user asks you to remember a preference (preference_text is the preference). "
    "out_of_scope: anything else (trivia, real-game talk, other leagues). "
    "Use thread history only to resolve short follow-ups. Return JSON only. No em dashes."
)


def _keyword_route(message: str) -> dict:
    m = (message or "").lower()
    base = {"player_names": [], "give_player_names": [], "receive_player_names": [], "picks": [],
            "other_team_hint": None, "metric_hint": None, "week": None, "preference_text": None}
    if re.search(r"\bremember\b|\bdon'?t forget\b|\bkeep in mind\b", m):
        return {**base, "intent": "remember", "preference_text": message.strip()}
    if re.search(r"\bwaiver|add\b|drop\b|faab|pick up\b", m):
        return {**base, "intent": "waivers"}
    if re.search(r"\bstart\b|\bsit\b|\bflex\b", m):
        return {**base, "intent": "start_sit"}
    if re.search(r"\btrade\b|\bswap\b", m):
        return {**base, "intent": "trade_eval"}
    if re.search(r"\bbuy\b|\bsell\b|\bhold\b", m):
        return {**base, "intent": "buy_sell_hold"}
    if re.search(r"what should i do|this week|my team", m):
        return {**base, "intent": "this_week"}
    if re.search(r"target share|snap share|route|metric|what is|who leads", m):
        hint = None
        for cand in ("target share", "snap share", "route rate", "opportunity share"):
            if cand in m:
                hint = cand
                break
        return {**base, "intent": "metric", "metric_hint": hint}
    return {**base, "intent": "out_of_scope"}


def classify(question: str, context: dict | None = None, history: list | None = None) -> dict:
    """Return {intent, entities{...}, preference_text}."""
    payload = {
        "question": question,
        "context": {k: (context or {}).get(k) for k in ("league_name", "team_name", "week", "season")},
        "history": (history or [])[-4:],
    }
    try:
        data = call_model_json(SYSTEM, json.dumps(payload, default=str), SCHEMA, "assistant_router")
        intent = data.get("intent") if data.get("intent") in INTENTS else "out_of_scope"
        entities = {
            "player_names": data.get("player_names") or [],
            "give_player_names": data.get("give_player_names") or [],
            "receive_player_names": data.get("receive_player_names") or [],
            "picks": data.get("picks") or [],
            "other_team_hint": data.get("other_team_hint"),
            "metric_hint": data.get("metric_hint"),
            "week": data.get("week"),
        }
        return {"intent": intent, "entities": entities, "preference_text": data.get("preference_text"), "router_model": "model", "prompt_version": ROUTER_PROMPT_VERSION}
    except Exception as exc:  # model down (rate limit / unavailable): keyword fallback
        logger.debug("[assistant] router model failed, keyword fallback: %s", exc)
        routed = _keyword_route(question)
        intent = routed.pop("intent")
        pref = routed.pop("preference_text")
        return {"intent": intent, "entities": routed, "preference_text": pref, "router_model": "keyword-fallback", "prompt_version": ROUTER_PROMPT_VERSION}
