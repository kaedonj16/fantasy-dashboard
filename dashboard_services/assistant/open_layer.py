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
from dashboard_services.assistant import compare_table as compare_table_mod
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
                    "max_age", "pool", "season", "players", "key",
                    "unsupported_key")

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
    "In a compare entry, a player's missing_metrics list names the "
    "numbers the data does not have for him: compare the two players on "
    "the metrics both of them have and name the missing ones for that "
    "player; never drop a player, or the comparison, over a missing "
    "metric. "
    "For a two-player compare entry, a side-by-side table of the two "
    "profiles renders with the answer and carries the numbers: keep the "
    "text to the basis and one short why paragraph, and do not list each "
    "player's numbers line by line. "
    "A rank_players entry carries a ranking computed by code: present "
    "the players in exactly that order, each with his ranking-key "
    "number, and never reorder them. A name in the entry's unresolved "
    "list was NOT ranked: say so by name (not found, offering the "
    "closest match when one is listed, or ambiguous between the listed "
    "candidates), and if the entry carries an unsupported_key, say "
    "plainly that the ranking is by value because that key is not "
    "tracked here. "
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


def _deterministic_compare_entry(bundle: dict, question: str) -> dict | None:
    """The compare_players entry for a two-player compare shape, or None.

    A compare shape (router.compare_player_names) whose two spans are
    both player-like (resolvable, ambiguous, or a near miss of a real
    name) is fetched deterministically, before the planner: the planner
    is a model and can fetch a single-player profile instead, collapsing
    the side-by-side into a one-sided answer (the live failure this
    seam closes). Any other shape returns None and the planner runs as
    usual, so pronoun follow-ups ("how does he compare to X") keep
    their thread-context path.
    """
    try:
        from dashboard_services.assistant.router import compare_player_names
    except Exception:
        return None
    spans = compare_player_names(question or "")
    if not spans:
        return None

    def _player_like(span: str) -> bool:
        pid, _info, ambiguous = tools_mod._resolve(bundle, span)
        if pid or ambiguous:
            return True
        try:
            from dashboard_services.assistant.resolve import nearest_names
            return bool(nearest_names(span, limit=1))
        except Exception:
            return False

    if not (_player_like(spans[0]) and _player_like(spans[1])):
        return None
    entry = tools_mod.run_tool("compare_players",
                               {"player_a": spans[0], "player_b": spans[1]},
                               bundle)
    if entry.get("ok") or entry.get("ambiguous") or entry.get("not_found"):
        return entry
    return None


def _deterministic_rank_entry(bundle: dict, question: str) -> dict | None:
    """The rank_players entry for a rank-a-named-set shape, or None.

    A rank shape (router.rank_shape) whose spans are player-like (at
    least two resolve, are ambiguous, or near-miss a real name) is
    fetched deterministically, before the planner: the planner is a
    model and can fetch one player's numbers and let the composer
    assert a ranking of the whole set from that subset (the collapse
    audit failure this seam closes). The entry is returned even when
    fewer than two names resolve, because it carries the named-gap
    detail the decline answers from. Any other shape returns None and
    the planner runs as usual.
    """
    try:
        from dashboard_services.assistant.router import rank_shape
    except Exception:
        return None
    shape = rank_shape(question or "")
    if not shape:
        return None

    def _player_like(span: str) -> bool:
        pid, _info, ambiguous = tools_mod._resolve(bundle, span)
        if pid or ambiguous:
            return True
        try:
            from dashboard_services.assistant.resolve import nearest_names
            return bool(nearest_names(span, limit=1))
        except Exception:
            return False

    if sum(1 for n in shape["names"] if _player_like(n)) < 2:
        return None
    return tools_mod.run_tool(
        "rank_players",
        {"players": list(shape["names"]), "key": shape["key"],
         "unsupported_key": shape.get("unsupported_key")}, bundle)


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
    if frozen_evidence is None:
        # A two-player compare shape fetches compare_players
        # deterministically and that entry IS the evidence: a resolved
        # pair compares side-by-side, an ambiguous name clarifies, and
        # a near miss declines by name with the closest match offered.
        compare_entry = _deterministic_compare_entry(bundle, question)
        if compare_entry is not None:
            pack["evidence"] = [compare_entry]
            pack["tools_called"] = ["compare_players"]
            pack["context_players"] = thread.get("context_players") or []
            if not compare_entry.get("ok"):
                pack["available"] = False
            return pack, ["compare_players"]
        # A rank-a-named-set shape fetches rank_players deterministically
        # and that entry IS the evidence: every named player resolved and
        # ordered by code, or a named-gap decline when fewer than two
        # resolve. The planner never improvises the set.
        rank_entry = _deterministic_rank_entry(bundle, question)
        if rank_entry is not None:
            pack["evidence"] = [rank_entry]
            pack["tools_called"] = ["rank_players"]
            pack["context_players"] = thread.get("context_players") or []
            if not rank_entry.get("ok"):
                pack["available"] = False
            return pack, ["rank_players"]
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


def _rank_decline_answer(entry: dict) -> dict:
    """Named decline for a rank ask where fewer than two players resolved.

    Every unresolved name is answered BY NAME (not found, with the
    closest real match offered; or ambiguous between named candidates),
    and the resolved side is stated, so the asker can re-ask with the
    right name. A generic decline here would hide which name failed,
    and a one-player "ranking" would be the subset failure again.
    """
    ranking = entry.get("ranking") or []
    resolved = [str(r.get("name")) for r in ranking if r.get("name")]
    parts = []
    follow = []
    for u in entry.get("unresolved") or []:
        asked = str(u.get("asked") or "").strip()
        if not asked:
            continue
        if u.get("reason") == "ambiguous":
            cands = ", ".join(str(c.get("name")) for c in
                              (u.get("candidates") or []) if c.get("name"))
            parts.append(f"{asked} matches more than one player"
                         + (f" ({cands})" if cands else "")
                         + ", so I cannot tell which one you mean.")
            continue
        parts.append(f"I could not find {asked} in the player pool.")
        near = [str(n) for n in (u.get("nearest") or []) if n]
        if near:
            parts.append(f"The closest match is {near[0]}.")
            if resolved and len(entry.get("unresolved") or []) == 1:
                ask = (f"Rank {', '.join(resolved + [near[0]])} by "
                       f"{entry.get('key_label') or 'value'}")
                follow.append(compose_mod.followup_pair(ask, ask))
    if len(resolved) == 1:
        parts.append(f"Only {resolved[0]} resolved, so there is no "
                     "ranking to make from this set.")
    elif not resolved:
        parts.append("None of those names resolved to a player in the "
                     "pool, so there is no ranking to make.")
    if entry.get("unsupported_key"):
        parts.append(f"{entry['unsupported_key']} is not a tracked "
                     "ranking key here, so rankings run by value.")
    return {"verdict": None, "text": " ".join(parts), "receipts": [],
            "caveat": None,
            "followups": compose_mod._clean_followups("open", follow),
            "act_link": None, "actions": [], "mentions": []}


def _fallback_verdict(evidence: list) -> str | None:
    for entry in evidence:
        ranking = entry.get("ranking") or []
        if entry.get("tool") == "rank_players" and ranking \
                and ranking[0].get("name"):
            return (f"{ranking[0]['name']} tops this group by "
                    f"{entry.get('key_label') or 'value'}")
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
        rank_entry = next((e for e in evidence
                           if e.get("tool") == "rank_players"), None)
        if rank_entry is not None:
            # A rank ask where fewer than two named players resolved:
            # decline BY NAME from the entry's gap detail instead of
            # the generic decline (or a one-player pseudo-ranking).
            return compose_mod._enforce_grounding(
                _rank_decline_answer(rank_entry), "open", pack, question)
        not_found = next((e for e in evidence if e.get("not_found")), None)
        if not_found is not None:
            # A named player could not be found: say so BY NAME and
            # offer the closest real match as a ready-made re-ask,
            # instead of the generic decline (or, worse, an answer about
            # only the player who did resolve).
            asked = str(not_found.get("not_found") or "").strip()
            near = [str(n) for n in (not_found.get("nearest") or []) if n]
            found_name = str((not_found.get("found") or {}).get("name")
                             or "").strip()
            text = (f"I could not find {asked} in the player pool."
                    if asked else
                    "I could not find one of those players in the player pool.")
            follow = []
            if near:
                text += f" The closest match is {near[0]}."
                if found_name:
                    ask = f"Compare {found_name} and {near[0]}"
                    follow.append(compose_mod.followup_pair(ask, ask))
            return compose_mod._enforce_grounding({
                "verdict": None, "text": text, "receipts": [],
                "caveat": None,
                "followups": compose_mod._clean_followups("open", follow),
                "act_link": None, "actions": [], "mentions": [],
            }, "open", pack, question)
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
    answer = compose_mod._enforce_grounding({
        "verdict": verdict,
        "text": text,
        "receipts": receipts,
        "caveat": None,
        "followups": compose_mod._clean_followups("open", draft.get("followups")),
        "act_link": link,
        "actions": [],
        "mentions": _evidence_mentions(usable),
    }, "open", pack, question)
    # The compare table rides the same answer dict as the verdict,
    # receipts and deep link: built deterministically from the compare
    # entry's profiles (never from the draft), so its numbers are the
    # pack's own and need no grounding pass.
    table = compare_table_mod.compare_table_for_evidence(usable)
    if table is not None:
        answer["compare_table"] = table
    return answer


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
