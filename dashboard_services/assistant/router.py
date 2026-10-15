"""Intent router: cheap model classification with a keyword fallback."""
from __future__ import annotations

import json
import logging
import re

from dashboard_services.ai.client import AIUnavailableError, AIRateLimitError
from dashboard_services.assistant import ROUTER_PROMPT_VERSION
from dashboard_services.assistant.model import call_model_json

logger = logging.getLogger(__name__)

INTENTS = ["start_sit", "trade_eval", "buy_sell_hold", "player", "metric", "leaderboard", "waivers", "this_week",
           "playoffs", "power_rankings", "standings", "matchup", "schedule",
           "remember", "out_of_scope"]

SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENTS},
        "player_names": {"type": "array", "items": {"type": "string"}},
        "give_player_names": {"type": "array", "items": {"type": "string"}},
        "receive_player_names": {"type": "array", "items": {"type": "string"}},
        "picks": {"type": "array", "items": {"type": "string"}},
        "other_team_hint": {"type": ["string", "null"]},
        "team_names": {"type": "array", "items": {"type": "string"}},
        "metric_hint": {"type": ["string", "null"]},
        "week": {"type": ["integer", "null"]},
        "preference_text": {"type": ["string", "null"]},
        "board_kind": {"type": ["string", "null"], "enum": ["board", "intersection", "dual", "movers", "matchup", None]},
        "second_metric_hint": {"type": ["string", "null"]},
        "top_n": {"type": ["integer", "null"]},
        "position": {"type": ["string", "null"]},
        "team": {"type": ["string", "null"]},
        "window_weeks": {"type": ["integer", "null"]},
        "direction": {"type": ["string", "null"], "enum": ["risers", "fallers", None]},
    },
    "required": ["intent", "player_names", "give_player_names", "receive_player_names", "picks", "other_team_hint", "team_names", "metric_hint", "week", "preference_text", "board_kind", "second_metric_hint", "top_n", "position", "team", "window_weeks", "direction"],
    "additionalProperties": False,
}

SYSTEM = (
    "You route fantasy football questions for the BR Fantasy assistant. "
    "Classify into exactly one intent and extract entities verbatim from the message. "
    "start_sit: who to start. trade_eval: a proposed trade with give and receive sides. "
    "buy_sell_hold: buy/sell/hold on a player. player: a general player question. "
    "metric: a question about one named player and a metric, or a metric definition "
    "(metric_hint is the metric name). "
    "leaderboard: a pool-wide ranking question with no single player as the subject: "
    "top N in a metric, who leads a metric, players ranked by a metric, players in the "
    "top N of two metrics (board_kind intersection, second_metric_hint is the second "
    "metric), one metric ranked with a second metric shown alongside (board_kind dual), "
    "biggest risers or fallers in a metric (board_kind movers, direction risers or "
    "fallers), or which defenses give up the most to a position (board_kind matchup, "
    "position is the offensive position). For leaderboard also extract top_n when a "
    "count is stated, position (QB/RB/WR/TE) and team (NFL abbreviation) filters when "
    "stated, and window_weeks when a last-N-weeks window is stated. "
    "waivers: adds, drops, or FAAB. this_week: what to do with my team this week. "
    "playoffs: playoff odds, clinching, elimination, or what a team needs to make the playoffs. "
    "power_rankings: the power rankings, who is strongest, or the team to beat; a ranking of team "
    "strength, NOT the win-loss standings table. "
    "standings: the standings table itself, records, points for, seeds, or division races; the "
    "win-loss table, NOT power rankings. "
    "matchup: a head-to-head matchup preview or a win probability between two teams this week "
    "(team_names holds the team names mentioned). "
    "schedule: remaining schedule strength, the easiest or hardest remaining schedule, or how "
    "hard a specific stretch of weeks is (a playoff stretch question is schedule, not playoffs). "
    "remember: the user asks you to remember a preference (preference_text is the preference). "
    "out_of_scope: anything else (trivia, real-game talk, other leagues). "
    "Use thread history only to resolve short follow-ups. Return JSON only. No em dashes."
)

_BOARD_ENTITY_DEFAULTS = {"board_kind": None, "second_metric_hint": None, "top_n": None,
                          "position": None, "team": None, "window_weeks": None,
                          "direction": None}


_METRIC_PHRASES = ("yards per route run", "air yards share", "target share",
                   "snap share", "opportunity share", "usage trend", "wopr",
                   "route rate", "yards per target", "yards per reception")

_POSITION_WORDS = (("quarterback", "QB"), ("running back", "RB"),
                   ("wide receiver", "WR"), ("tight end", "TE"),
                   (" qbs", "QB"), (" rbs", "RB"), (" wrs", "WR"), (" tes", "TE"),
                   (" qb", "QB"), (" rb", "RB"), (" wr", "WR"), (" te", "TE"))


def _metric_hints_in(m: str) -> list[str]:
    # Text order: the first metric named is the primary board.
    return sorted((cand for cand in _METRIC_PHRASES if cand in m), key=m.find)


def _keyword_board(message: str, m: str) -> dict | None:
    """Pool-wide ranking phrasings -> the leaderboard intent (or None)."""
    hints = _metric_hints_in(m)
    top = re.search(r"\btop\s+(\d{1,2})\b", m)
    window = re.search(r"\blast\s+(\d{1,2})\s+weeks?\b", m)
    position = None
    for word, pos in _POSITION_WORDS:
        if word in m:
            position = pos
            break
    kind = None
    direction = None
    if re.search(r"gives? up the most|allow(s|ed)? the most|worst defenses?|easiest defenses?", m):
        kind = "matchup"
    elif re.search(r"\brisers?\b|\brising\b|heating up|trending up", m):
        kind, direction = "movers", "risers"
    elif re.search(r"\bfallers?\b|\bfalling\b|cooling off|trending down", m):
        kind, direction = "movers", "fallers"
    elif len(hints) >= 2 and top:
        kind = "intersection"
    elif hints and (top or re.search(r"\bwho (leads|has)\b|\bleaderboard\b|\branked\b|\bmost\b.{0,30}\bshare\b", m)):
        kind = "board"
    if kind is None:
        return None
    return {
        "intent": "leaderboard",
        "metric_hint": hints[0] if hints else None,
        "second_metric_hint": hints[1] if len(hints) > 1 else None,
        "board_kind": kind,
        "top_n": int(top.group(1)) if top else None,
        "position": position,
        "team": None,
        "window_weeks": int(window.group(1)) if window else None,
        "direction": direction,
    }


def _keyword_route(message: str) -> dict:
    m = (message or "").lower()
    base = {"player_names": [], "give_player_names": [], "receive_player_names": [], "picks": [],
            "other_team_hint": None, "team_names": [], "metric_hint": None, "week": None,
            "preference_text": None, **_BOARD_ENTITY_DEFAULTS}
    if re.search(r"\bremember\b|\bdon'?t forget\b|\bkeep in mind\b", m):
        return {**base, "intent": "remember", "preference_text": message.strip()}
    # League-context intents run before the player/team-week rules below:
    # their phrasings overlap ("this week", "my team", "who leads"), so the
    # specific subject wins. Disambiguation: power rankings need the word
    # power (or team-to-beat / strongest), standings own the record table
    # (standings, division, points for, place), and a matchup mention beats
    # the generic this-week digest.
    if re.search(r"playoff stretch", m):
        # "Playoff stretch" names the schedule window (weeks 15-17), not the
        # odds question; it is the one phrase that outranks the playoff rule.
        return {**base, "intent": "schedule"}
    if re.search(r"playoff|postseason|clinch|eliminat", m):
        return {**base, "intent": "playoffs"}
    if re.search(r"power ranking|power rank|team to beat|strongest team", m):
        return {**base, "intent": "power_rankings"}
    if re.search(r"standings|division|most points for|first place|what place am i", m):
        return {**base, "intent": "standings"}
    if re.search(r"\bmatchup\b|who wins between|win probability|win prob\b", m):
        return {**base, "intent": "matchup"}
    if re.search(r"remaining schedule|schedule strength|easiest schedule|hardest schedule|"
                 r"toughest schedule|playoff stretch|my schedule|league schedule", m):
        return {**base, "intent": "schedule"}
    if re.search(r"\bwaiver|add\b|drop\b|faab|pick up\b", m):
        return {**base, "intent": "waivers"}
    board = _keyword_board(message, m)
    if board is not None:
        return {**base, **board}
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
            "team_names": data.get("team_names") or [],
            "metric_hint": data.get("metric_hint"),
            "week": data.get("week"),
            "board_kind": data.get("board_kind"),
            "second_metric_hint": data.get("second_metric_hint"),
            "top_n": data.get("top_n"),
            "position": data.get("position"),
            "team": data.get("team"),
            "window_weeks": data.get("window_weeks"),
            "direction": data.get("direction"),
        }
        return {"intent": intent, "entities": entities, "preference_text": data.get("preference_text"), "router_model": "model", "prompt_version": ROUTER_PROMPT_VERSION}
    except Exception as exc:  # model down (rate limit / unavailable): keyword fallback
        logger.debug("[assistant] router model failed, keyword fallback: %s", exc)
        routed = _keyword_route(question)
        intent = routed.pop("intent")
        pref = routed.pop("preference_text")
        return {"intent": intent, "entities": routed, "preference_text": pref, "router_model": "keyword-fallback", "prompt_version": ROUTER_PROMPT_VERSION}
