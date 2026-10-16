"""Open answering path: questions the fixed intents do not cover.

The fixed intents stay the pinned path for verdict questions (trades,
start/sit, keepers). Everything else the site's data can answer routes
here when routing finds no confident fixed intent: the model plans a
small set of read-only tool calls (dashboard_services.assistant.tools),
code fetches them deterministically, and the answer composes from the
fetched evidence only, under the same grounding parity gate as the pack
path (compose._enforce_grounding over the evidence pack). Zero invented
numbers is the bar here too: a number that is not in a tool result or
the question never ships.

Bounds: at most MAX_TOOL_CALLS tool calls across at most MAX_PLANNER
ROUNDS planning rounds, with a wall-clock deadline; when the deadline
or the call budget runs out, the answer composes from whatever evidence
exists, and when no evidence exists the answer declines honestly
instead of guessing. The planner model failing with zero evidence
raises like the pack path does, so the route can 503 the same way.

Thread context: the planner receives the thread's already-resolved
players (from the router's entities on earlier turns) plus the last
answer, so follow-ups ("what about his handcuff?", "and how does he
compare to Addison?") resolve against the conversation, not cold.
"""
from __future__ import annotations

import json
import logging
import time

from dashboard_services.assistant import aliases, tools as tools_mod
from dashboard_services.assistant import compose as compose_mod
from dashboard_services.assistant import evidence as ev
from dashboard_services.assistant.model import call_model_json

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS = 4
MAX_PLANNER_ROUNDS = 2
DEADLINE_S = 25.0

_PLANNER_SYSTEM = (
    "You plan data fetches for the BR Fantasy assistant. The user asked a "
    "question the fixed question types did not claim. Choose the smallest "
    "set of tool calls that fetches everything needed to answer it from "
    "the site's own data. When the question names a season (in 2024, "
    "last season), pass that season to the board tools; a board fetched "
    "without a season covers the current season only and must never be "
    "presented as another season's. The payload lists the available tools with "
    "their arguments, the league context, and thread_context: players "
    "already resolved earlier in this conversation. When the question "
    "says him, his, he, or them, it refers to those thread players. A "
    "question with two parts gets one call per part. Use the exact "
    "player and team names from the question or the thread context. "
    "If no tool can fetch what the question needs (real-world trivia, "
    "weather, other sports, news outside the site), return decline true "
    "with no calls. Return JSON only. No em dashes."
)

_PLAN_ARG_FIELDS = ("player", "player_a", "player_b", "stat", "metric",
                    "position", "team", "kind", "top_n", "family",
                    "max_age", "pool", "season")

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "tool_calls": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "tool": {"type": "string",
                         "enum": [t["name"] for t in tools_mod.TOOL_CATALOG]},
                **{f: {"type": ["string", "null"]} for f in _PLAN_ARG_FIELDS
                   if f not in ("top_n", "max_age")},
                "top_n": {"type": ["integer", "null"]},
                "max_age": {"type": ["integer", "null"]},
            },
            "required": ["tool", *_PLAN_ARG_FIELDS],
            "additionalProperties": False}},
        "decline": {"type": "boolean"},
    },
    "required": ["tool_calls", "decline"],
    "additionalProperties": False,
}

_OPEN_SYSTEM = (
    "You are the BR Fantasy assistant. Answer the user's question from "
    "the fetched site data in the evidence ONLY. Every number you state "
    "must appear verbatim in the evidence; never use a number from "
    "memory. When evidence carries a basis line (season totals, a week "
    "window, a qualified pool), state that basis in the answer so the "
    "number is never mistaken for a different one. Receipts: 2 to 4 "
    "entries, each a label and an evidence value as a string. If "
    "evidence marks gated metrics or a gated board, say that part is a "
    "PRO metric on BR Fantasy and give no numbers for it; answer the "
    "rest from what is there. If the evidence answers only part of the "
    "question, answer that part and say plainly what is not in the data. "
    "Verdict: a short headline when the evidence supports one, else "
    "null. Never mention tools, evidence, or any internal data "
    "structure; call it your league data or the site's numbers. "
    "Follow-ups are objects with a label and an ask: the label is the "
    "short chip text, the ask is the exact message sent when tapped, "
    "written as a natural question in the user's own voice. No em "
    "dashes. Use plain hyphens. Short, direct sentences. Return JSON only."
)


# -- probe: does this question touch site data at all? ---------------------------

def _metric_anchor(question: str) -> str | None:
    """A catalog metric key named in the question (by key or label)."""
    norm = " " + (question or "").lower().replace("_", " ") + " "
    try:
        config = ev._metric_config()
    except Exception:
        return None
    for key, spec in config.items():
        label = str((spec or {}).get("label") or "").lower()
        if f" {key.replace('_', ' ')} " in norm or (label and f" {label} " in norm):
            return key
    return None


def probe(question: str, bundle: dict, last_entities: dict | None = None) -> dict:
    """Deterministic site-data anchors for a question: resolved players
    (from the question and the thread), league team names, a stat
    phrase, a catalog metric. The open path runs only when at least one
    anchor exists, so plain trivia keeps the out-of-scope decline and
    never spends a model call."""
    from dashboard_services.assistant.resolve import resolve_player
    from dashboard_services.assistant.router import _player_name_in

    names: list[str] = []
    span = _player_name_in(question or "")
    if span:
        names.append(span)
    for n in (last_entities or {}).get("player_names") or []:
        if n and n not in names:
            names.append(str(n))
    players = []
    roster_ids = (bundle or {}).get("roster_player_ids") or set()
    for name in names:
        r = resolve_player(name, roster_player_ids=roster_ids)
        if r["status"] == "resolved":
            players.append({"name": name, "player_id": r["player_id"]})
    teams = []
    low = (question or "").lower()
    for _rid, team_name, _roster in tools_mod._team_rows(bundle or {}):
        if team_name and str(team_name).lower() in low:
            teams.append(str(team_name))
    stat = aliases.resolve_phrase(question or "")
    metric_key = _metric_anchor(question or "")
    anchors = bool(players or teams or stat or metric_key)
    return {"anchors": anchors, "players": players, "teams": teams,
            "stat": stat, "metric_key": metric_key}


# -- planner + fetch --------------------------------------------------------------

def _plan_args(call: dict) -> dict:
    return {f: call.get(f) for f in _PLAN_ARG_FIELDS if call.get(f) is not None}


def _frozen_lookup(frozen: list, used: set, tool: str) -> dict:
    for i, entry in enumerate(frozen):
        if i not in used and isinstance(entry, dict) \
                and entry.get("tool") == tool:
            used.add(i)
            return entry
    return {"tool": tool, "ok": False, "missing": "no fetched result for that tool",
            "players": [], "receipts": [], "link": None}


def plan_and_fetch(question: str, bundle: dict, thread: dict | None = None,
                   frozen_evidence: list | None = None) -> tuple[dict, list[str]]:
    """Run the planner loop and build the open evidence pack.

    Returns (pack, tools_called). The pack is shaped like a builder pack
    (type/league/week/as_of) plus "evidence": the tool entries, and
    "context_players": the thread's resolved players the planner saw.
    With frozen_evidence (the eval seam), planner calls are served from
    the frozen entries by tool name instead of executing live tools.
    """
    thread = thread or {}
    pack = ev._base_pack("open", bundle)
    started = time.monotonic()
    evidence: list[dict] = []
    tools_called: list[str] = []
    seen_calls: set[str] = set()
    frozen_used: set[int] = set()
    declined = False
    payload_base = {
        "question": question,
        "league": {"name": bundle.get("league_name"),
                   "team_name": bundle.get("team_name"),
                   "week": bundle.get("week"), "season": bundle.get("season")},
        "thread_context": {
            "players": thread.get("context_players") or [],
            "last_intent": thread.get("last_intent"),
            "last_answer": (thread.get("last_answer") or "")[:400] or None,
        },
        "tools": tools_mod.TOOL_CATALOG,
    }
    for round_no in range(MAX_PLANNER_ROUNDS):
        if len(tools_called) >= MAX_TOOL_CALLS:
            break
        if round_no > 0 and time.monotonic() - started > DEADLINE_S:
            break
        payload = dict(payload_base)
        payload["round"] = round_no + 1
        if evidence:
            payload["results_so_far"] = evidence
        try:
            plan = call_model_json(_PLANNER_SYSTEM, json.dumps(payload, default=str),
                                   _PLAN_SCHEMA, "assistant_open_planner")
        except Exception:
            if evidence:
                logger.info("[assistant] open planner failed mid-loop; "
                            "composing from %d fetched entries", len(evidence))
                break
            raise
        if plan.get("decline") and not evidence:
            declined = True
            break
        calls = plan.get("tool_calls") or []
        if not calls:
            break
        progressed = False
        for call in calls:
            if len(tools_called) >= MAX_TOOL_CALLS:
                break
            if time.monotonic() - started > DEADLINE_S:
                break
            if not isinstance(call, dict) or not call.get("tool"):
                continue
            name = str(call["tool"])
            args = _plan_args(call)
            dedupe = name + "|" + json.dumps(args, sort_keys=True, default=str)
            if dedupe in seen_calls:
                continue
            seen_calls.add(dedupe)
            if frozen_evidence is not None:
                entry = _frozen_lookup(frozen_evidence, frozen_used, name)
            else:
                entry = tools_mod.run_tool(name, args, bundle)
            evidence.append(entry)
            tools_called.append(name)
            progressed = True
        if not progressed:
            break
    pack["evidence"] = evidence
    pack["tools_called"] = tools_called
    pack["context_players"] = thread.get("context_players") or []
    if declined:
        pack["declined"] = True
    if not any(e.get("ok", True) and not e.get("gated") for e in evidence):
        # Nothing usable was fetched: keep the pack honest so the
        # composer declines instead of improvising.
        pack["available"] = False
    return pack, tools_called


# -- composition --------------------------------------------------------------------

def decline_answer(nearest=None) -> dict:
    """The honest open-path decline, in the backend's no-data shape."""
    text = "That is not available from your league data."
    followups = []
    if nearest:
        text += (f" The nearest tracked options are "
                 f"{', '.join(str(n) for n in nearest[:3])}. Ask about one "
                 f"of those and I can pull the live numbers.")
        followups = [compose_mod.followup_pair(f"What is {n}?")
                     for n in nearest[:2]]
    if not followups:
        followups = [dict(p) for p in
                     compose_mod._FOLLOWUP_TEMPLATES.get("open", [])][:2]
    return {"verdict": None, "text": text, "receipts": [], "caveat": None,
            "followups": followups, "act_link": None, "actions": [],
            "mentions": []}


def _gated_open_answer(entry: dict) -> dict:
    """Deterministic PRO-gated answer for a gated open fetch: parity
    with the site's gated surfaces, no numbers anywhere."""
    label = entry.get("label") or "That metric"
    text = (f"{label} is a PRO metric on BR Fantasy, so its numbers are not "
            "included on the free tier. Free boards I can pull right now "
            "include target share, air yards share, and snap share.")
    return {"verdict": None, "text": text, "receipts": [], "caveat": None,
            "followups": [compose_mod.followup_pair("Who leads in target share?"),
                          compose_mod.followup_pair("Who leads in snap share?")],
            "act_link": {"href": "/metrics", "label": "View on Advanced Metrics"},
            "actions": [], "mentions": []}


def _fallback_verdict(evidence: list) -> str | None:
    for entry in evidence:
        board = entry.get("board") or []
        if board and board[0].get("name"):
            return f"{board[0]['name']} leads {entry.get('label') or 'the board'}"
    for entry in evidence:
        player = entry.get("player") or {}
        if player.get("name"):
            return f"{player['name']} update"
    for entry in evidence:
        if entry.get("label") and entry.get("definition"):
            return str(entry["label"])
    return None


def _receipt_match_keys(value) -> set:
    """Canonical keys under which an evidence value can be cited in prose.

    The direct canonical form, plus the percent form for fractions: a
    share stored as 0.29 is naturally stated as "29%" in prose, and the
    two must recognize each other when matching a citation to its
    source row. Matching only: the receipt itself always carries the
    evidence's own value, never a rescaled one.
    """
    keys = {compose_mod._canon(str(value))}
    try:
        f = float(compose_mod._norm_num(str(value)))
    except (TypeError, ValueError):
        f = None
    if f is not None and 0 < abs(f) < 1:
        keys.add(compose_mod._canon(str(abs(f) * 100)))
    return keys


def _complete_citations(verdict, text, model_receipts, evidence) -> list:
    """Citation completion: every number the final verdict+text states
    gets a receipt row sourced from the tool evidence that produced it.

    The parity gate proves a stated number is grounded in the evidence,
    but the model's own receipt picks are independent of its prose, so
    a number the draft leaned on could ship with no receipt behind it
    (live eval round 3: ow4, ow9, ow13, ow15, ow17 all failed on
    uncited key numbers with zero invented). Rows are taken from the
    evidence entries' own receipt rows (label and value as the tools
    built them), prioritized: rows backing a stated number first, then
    the model's surviving receipts, then any remaining evidence rows,
    deduplicated by value and capped at the standard 4. A stated number
    with no evidence source gets NO row here: the grounding gate strips
    it from the prose, and no receipt is ever invented for it.
    """
    cited = compose_mod._canon_numbers(verdict) \
        | compose_mod._canon_numbers(text)
    backed, generic = [], []
    for entry in evidence:
        for r in entry.get("receipts") or []:
            if not (isinstance(r, dict) and r.get("label")
                    and r.get("value") not in (None, "")):
                continue
            row = {"label": str(r["label"]), "value": str(r["value"])}
            if _receipt_match_keys(r["value"]) & cited:
                backed.append(row)
            else:
                generic.append(row)
    out, seen = [], set()
    for row in backed + list(model_receipts or []) + generic:
        if not isinstance(row, dict) or row.get("value") in (None, ""):
            continue
        key = compose_mod._canon(str(row["value"]))
        if key in seen:
            continue
        seen.add(key)
        out.append({"label": str(row.get("label") or ""),
                    "value": str(row["value"])})
        if len(out) >= 4:
            break
    return out


def _evidence_mentions(evidence: list) -> list:
    found, seen = [], set()
    for entry in evidence:
        for p in entry.get("players") or []:
            if isinstance(p, dict) and p.get("player_id") and p.get("name") \
                    and str(p["player_id"]) not in seen:
                seen.add(str(p["player_id"]))
                found.append({"player_id": str(p["player_id"]),
                              "name": str(p["name"])})
    return found[:8]


def compose_open_answer(question: str, pack: dict, *, history=None,
                        preferences=None, assumed_context=None) -> dict:
    """Compose the standard answer shape from an open evidence pack.

    The pack's evidence is the only grounding set: the draft passes
    through compose's grounding retry, and the final answer passes the
    same parity gate as every pack answer. Gated-only evidence gets the
    deterministic gated answer; ambiguity gets the clarify shape;
    empty evidence gets the honest decline. No model call happens in
    those three cases.
    """
    evidence = [e for e in ((pack or {}).get("evidence") or [])
                if isinstance(e, dict)]
    usable = [e for e in evidence if e.get("ok", True) and not e.get("gated")]
    if not usable:
        gated = next((e for e in evidence if e.get("gated")), None)
        if gated is not None:
            return compose_mod._enforce_grounding(
                _gated_open_answer(gated), "open", pack, question)
        ambiguous = next((e for e in evidence if e.get("ambiguous")), None)
        if ambiguous is not None:
            return {"verdict": None, "text": "Which player do you mean?",
                    "receipts": [], "caveat": None, "followups": [],
                    "act_link": None, "actions": [], "mentions": [],
                    "candidates": ambiguous.get("ambiguous") or []}
        nearest = []
        for e in evidence:
            for n in e.get("nearest") or []:
                if n not in nearest:
                    nearest.append(n)
        return compose_mod._enforce_grounding(
            decline_answer(nearest), "open", pack, question)
    payload = {
        "intent": "open",
        "question": question,
        "evidence_pack": pack,
        "preferences": [p.get("text") if isinstance(p, dict) else str(p)
                        for p in (preferences or [])],
        "assumed_context": assumed_context or {},
        "history": (history or [])[-3:],
    }
    allowed = compose_mod._ground_set(pack, question)
    data = compose_mod._draft(_OPEN_SYSTEM, payload, allowed)
    verdict = data.get("verdict")
    if not (verdict and str(verdict).strip()):
        verdict = _fallback_verdict(usable)
    # Required mentions, pack-path machinery reused directly: the teams
    # the fetched evidence is about must be named in the prose (live
    # eval round 3 ow12 named neither). _required_mentions has an
    # "open" branch that reads the teams off the evidence entries, so
    # the injection is the same deterministic one the packs get.
    draft = {"verdict": verdict, "text": data.get("text") or "",
             "caveat": None, "followups": data.get("followups")}
    draft = compose_mod._ensure_required_mentions("open", pack, draft, [])
    verdict = draft.get("verdict")
    text = draft.get("text") or ""
    receipts = _complete_citations(
        verdict, text, compose_mod._final_receipts(data, allowed), usable)
    link = None
    for entry in usable:
        if entry.get("link"):
            link = entry["link"]
            break
    return compose_mod._enforce_grounding({
        "verdict": verdict,
        "text": text,
        "receipts": receipts,
        "caveat": None,
        "followups": compose_mod._clean_followups("open", draft.get("followups")),
        "act_link": link,
        "actions": [],
        "mentions": _evidence_mentions(usable),
    }, "open", pack, question)


def answer_open(question: str, bundle: dict, *, thread: dict | None = None,
                history=None, preferences=None, assumed_context=None,
                frozen_evidence: list | None = None):
    """Full open turn for the service: plan, fetch, compose.

    Returns (answer, pack, tools_called, player_names): player_names are
    the names the evidence resolved, stored as the thread's entities so
    the next turn's follow-ups inherit them.
    """
    pack, tools_called = plan_and_fetch(
        question, bundle, thread=thread, frozen_evidence=frozen_evidence)
    answer = compose_open_answer(question, pack, history=history,
                                 preferences=preferences,
                                 assumed_context=assumed_context)
    names, seen = [], set()
    for entry in pack.get("evidence") or []:
        for p in entry.get("players") or []:
            if isinstance(p, dict) and p.get("name") \
                    and str(p["name"]) not in seen:
                seen.add(str(p["name"]))
                names.append(str(p["name"]))
    return answer, pack, tools_called, names
