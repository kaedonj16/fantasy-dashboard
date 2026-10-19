"""Grounding scorer for the Stage 1 assistant eval (workshop section 5).

Pure functions, no model calls, no backend imports.

Scoring dimensions per answer:
- grounding: every number in the answer (verdict, text, caveat, receipts,
  follow-ups, action rows) must appear in the frozen evidence pack or in the
  question's own text. Anything else is "invented" and fails the answer.
- verdict: the call matches the frozen reference (per-type rules below).
- traps: the missing-stat question must come back in the backend's no-data
  shape with no invented number; the namesake question must resolve to one
  of the accepted outcomes in expect.accept.
- structure: verdict first (non-empty verdict) plus 2 to 4 receipts when the
  pack carries enough numbers; waived for no-data and clarify outcomes.

Number normalization: "$1,250" -> "1250", "17.90" -> "17.9", "24%" -> "24".
Sign is not significant ("-20.1" in a pack grounds "20.1" in prose, since
answers routinely state a gap or a drop as a magnitude). Numbers embedded in
words or codes (WR1, W7, TE12) are not numeric tokens and are ignored on
both sides, so a positional label never counts as an invented number.
"""
from __future__ import annotations

import json
import re

# Standalone numeric token: optional $ / sign, digits with optional comma
# grouping and decimals, optional trailing %. The lookarounds reject digits
# glued to letters or digits on either side (WR1, W7, 2026season).
_TOKEN = re.compile(r"(?<![A-Za-z0-9])\$?-?\d[\d,]*(?:\.\d+)?%?(?![A-Za-z0-9])")


def _canon(raw: str) -> str | None:
    s = str(raw).replace("$", "").replace(",", "").rstrip("%").strip()
    if not s:
        return None
    try:
        f = abs(float(s))
    except (TypeError, ValueError):
        return None
    if f == int(f):
        return str(int(f))
    return str(f)


def extract_numbers(text) -> list[str]:
    """Normalized numeric tokens in *text*, in order of appearance."""
    if text is None:
        return []
    if not isinstance(text, str):
        text = str(text)
    out = []
    for m in _TOKEN.finditer(text):
        c = _canon(m.group(0))
        if c is not None:
            out.append(c)
    return out


def pack_numbers(pack) -> set[str]:
    """Every numeric value in the pack JSON, normalized like extract_numbers.

    Numbers inside pack strings (gate receipts, headlines) count too: they
    are values the builders put there for the answer to cite.
    """
    if pack is None:
        return set()
    return set(extract_numbers(json.dumps(pack, default=str)))


def _answer_texts(answer: dict) -> list[str]:
    """Every answer surface a reader can see numbers on."""
    texts = []
    if not isinstance(answer, dict):
        return [str(answer or "")]
    for key in ("verdict", "text", "caveat"):
        if answer.get(key):
            texts.append(str(answer[key]))
    for r in answer.get("receipts") or []:
        if isinstance(r, dict):
            texts.append(str(r.get("label") or ""))
            texts.append(str(r.get("value") or ""))
    for f in answer.get("followups") or []:
        if isinstance(f, dict):
            # Pair shape: the label is reader-visible and the ask is
            # what gets sent, so numbers on either surface count.
            texts.append(str(f.get("label") or ""))
            texts.append(str(f.get("ask") or ""))
        else:
            texts.append(str(f))
    for a in answer.get("actions") or []:
        if isinstance(a, dict):
            texts.append(str(a.get("headline") or ""))
            texts.append(str(a.get("detail") or ""))
    return texts


def answer_numbers(answer: dict) -> set[str]:
    out: set[str] = set()
    for t in _answer_texts(answer):
        out.update(extract_numbers(t))
    return out


def _full_text(answer: dict) -> str:
    return " ".join(_answer_texts(answer))


def _canon_key_numbers(key_numbers) -> list[str]:
    out = []
    for n in key_numbers or []:
        c = _canon(n)
        if c is not None:
            out.append(c)
    return out


# -- verdict helpers -----------------------------------------------------------

def _trade_bucket(text: str) -> str | None:
    t = (text or "").upper()
    if "ACCEPT" in t or re.search(r"\bTAKE IT\b|\bTAKE THE\b", t):
        return "ACCEPT"
    if "DECLINE" in t or "REJECT" in t or "PASS ON" in t:
        return "DECLINE"
    if "FAIR" in t or "EVEN" in t or "BALANCED" in t:
        return "FAIR"
    return None


def _bsh_bucket(text: str) -> str | None:
    t = (text or "").upper()
    for word in ("BUY", "SELL", "HOLD"):
        if re.search(rf"\b{word}\b", t):
            return word
    return None


def _is_clarify(answer: dict) -> bool:
    if not isinstance(answer, dict):
        return False
    if answer.get("candidates"):
        return True
    if str(answer.get("outcome") or "").lower() == "clarify":
        return True
    verdict = answer.get("verdict")
    text = (answer.get("text") or "").lower()
    return verdict in (None, "") and "which" in text and "mean" in text


def _namesake_outcome(question: dict, answer: dict) -> str | None:
    """'clarify', 'pick:<name>', or None for the namesake trap question."""
    if _is_clarify(answer):
        return "clarify"
    verdict = str((answer or {}).get("verdict") or "")
    text = str((answer or {}).get("text") or "")
    pack = question.get("evidence_pack") or {}
    names = [p.get("name") for p in pack.get("players") or [] if isinstance(p, dict) and p.get("name")]
    for name in names:
        if name.lower() in verdict.lower():
            return f"pick:{name}"
    # A pick stated only in prose still counts, but only for the rostered
    # engine pick; anything else is an unaccepted guess between namesakes.
    pick = (question.get("reference") or {}).get("pick")
    if pick and pick.lower() in text.lower() and "start" in text.lower():
        return f"pick:{pick}"
    return None


# The unavailability declarations compose._missing_text and the
# decline texts can emit: an honest no-data answer states one of these
# with no verdict. The list mirrors the production translations; when a
# new translation lands, its declaration joins this list.
_UNAVAILABILITY_PHRASES = (
    "not available", "not tracked", "could not", "cannot", "do not have",
    "does not have", "no scheduled", "do not apply", "not enough",
    "signed-in", "no one behind",
)


def _no_data_shape(answer: dict) -> bool:
    """The backend's no-data patterns (compose.no_data_answer and the
    deterministic empty value board). Grounding is enforced separately
    on every answer, so accepting a shape here never lets an invented
    number through; this only recognizes the honest-empty forms."""
    if not isinstance(answer, dict):
        return False
    text = (answer.get("text") or "").lower()
    if answer.get("verdict") in (None, "") and any(
            phrase in text for phrase in _UNAVAILABILITY_PHRASES):
        return True
    # Empty buy-low / sell-high board (compose._empty_value_board_answer):
    # a "No ..." verdict stating the board is empty, prose that says no
    # candidate clears the screen and names the closest miss. This shape
    # is only reachable when the pack itself says no data.
    verdict = (answer.get("verdict") or "").lower()
    return verdict.startswith("no ") and "clears the screen" in text


# -- compare table (ow22 family) ------------------------------------------------
# The expected block is recomputed here from the frozen pack's compare
# profiles, independently of the production builder: same grouped rows
# in the same order, house-form values (shares as percents), the winner
# on each row, and the gap as the delta. A metric missing on either
# profile must have NO row; a tied row has no winner and no delta.

_COMPARE_GROUPS = (
    ("PRODUCTION", (
        ("ppg", "Points per game", "points"),
        ("recent_ppg", "Recent points per game", "points"),
        ("expected_ppg", "Expected points per game", "points"),
    )),
    ("OPPORTUNITY", (
        ("touches_per_game", "Touches per game", "points"),
        ("target_share", "Target share", "share"),
        ("snap_share", "Snap share", "share"),
    )),
    ("VALUE", (
        ("value", "Dynasty value", "points"),
    )),
)


def _house(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return int(f) if f == int(f) else round(f, 1)


def _expected_compare_table(detail):
    a, b = detail
    groups = []
    for group_label, metrics in _COMPARE_GROUPS:
        rows = []
        for key, label, kind in metrics:
            na, nb = _house(a.get(key)), _house(b.get(key))
            if na is None or nb is None:
                continue
            winner, delta = None, None
            if na != nb:
                winner = 0 if na > nb else 1
                delta = "+" + str(_house(abs(na - nb)))
            rows.append({"key": key, "label": label,
                         "a": str(na) + ("%" if kind == "share" else ""),
                         "b": str(nb) + ("%" if kind == "share" else ""),
                         "winner": winner, "delta": delta})
        if rows:
            groups.append({"label": group_label, "rows": rows})
    return groups


def _check_compare_table(question: dict, answer: dict):
    pack = question.get("evidence_pack") or {}
    entry = next((e for e in (pack.get("evidence") or [])
                  if isinstance(e, dict) and e.get("tool") == "compare_players"
                  and e.get("ok") and len(e.get("players_detail") or []) == 2),
                 None)
    if entry is None:
        return False, ["compare_table expected but the pack has no usable "
                       "two-player compare entry"]
    detail = entry["players_detail"]
    block = (answer or {}).get("compare_table")
    if not isinstance(block, dict):
        return False, ["compare_table block missing from the answer"]
    notes = []
    names = [str(p.get("name") or "") for p in block.get("players") or []]
    want_names = [str(p.get("name") or "") for p in detail]
    if names != want_names:
        notes.append(f"compare_table players {names} != pack profiles {want_names}")
    if block.get("groups") != _expected_compare_table(detail):
        notes.append("compare_table groups do not match the block recomputed "
                     "from the pack profiles (rows, values, winners, deltas)")
    return not notes, notes


def _meaningful_pack_numbers(question: dict) -> set[str]:
    """Pack numbers minus league/week/season boilerplate."""
    pack = question.get("evidence_pack") or {}
    boiler = set()
    for v in (pack.get("week"), (pack.get("as_of") or {}).get("week"),
              (pack.get("as_of") or {}).get("season"), (pack.get("league") or {}).get("season")):
        c = _canon(v) if v is not None else None
        if c is not None:
            boiler.add(c)
    return pack_numbers(pack) - boiler


# -- main entry ------------------------------------------------------------------

def score_answer(question: dict, answer: dict) -> dict:
    """Score one answer against its frozen question. See module docstring."""
    notes: list[str] = []
    pack = question.get("evidence_pack") or {}
    reference = question.get("reference") or {}
    expect = question.get("expect") or {}
    trap = question.get("trap") or {}
    qtype = question.get("type")

    # Grounding
    allowed = pack_numbers(pack) | set(extract_numbers(question.get("question") or ""))
    found = answer_numbers(answer)
    invented = sorted(found - allowed, key=lambda s: float(s))
    grounding_ok = not invented
    if invented:
        notes.append(f"invented numbers: {', '.join(invented)}")

    full = _full_text(answer)
    verdict_text = str((answer or {}).get("verdict") or "")
    no_data_expected = bool(expect.get("no_data")) or (isinstance(pack, dict) and pack.get("available") is False)
    gated_expected = isinstance(pack, dict) and bool(pack.get("gated"))
    clarify = _is_clarify(answer)

    # Verdict
    verdict_ok = False
    if trap.get("kind") == "namesake":
        outcome = _namesake_outcome(question, answer)
        accept = expect.get("accept") or []
        verdict_ok = outcome is not None and outcome in accept
        notes.append(f"namesake outcome: {outcome or 'unresolved'} (accept: {accept})")
    elif expect.get("gated"):
        # PRO-gate: no verdict, no board numbers, and the answer names PRO as
        # the reason (parity with the site's gated surfaces).
        verdict_ok = not verdict_text.strip() and "pro" in full.lower()
        if not verdict_ok:
            notes.append("expected the PRO-gated answer shape (no verdict, PRO named)")
    elif gated_expected:
        # A PRO-gated pack carries no findings by construction: the answer
        # passes only if it states the gate and shows no receipts/findings.
        verdict_ok = "pro" in full.lower() and not (answer or {}).get("receipts")
        if not verdict_ok:
            notes.append("gated pack: the answer must state the PRO gate and carry no receipts")
    elif no_data_expected:
        verdict_ok = _no_data_shape(answer)
        if not verdict_ok:
            notes.append("expected the no-data answer shape (no verdict, unavailability stated)")
    elif qtype == "start_sit":
        pick = reference.get("pick") or ""
        if pick and pick.lower() in verdict_text.lower():
            verdict_ok = True
        elif reference.get("close_call") and (answer or {}).get("caveat"):
            pack_names = [p.get("name") or "" for p in pack.get("players") or [] if isinstance(p, dict)]
            if any(n and n.lower() in verdict_text.lower() for n in pack_names):
                verdict_ok = True
                notes.append("close call: the other side is acceptable with a caveat")
        if not verdict_ok:
            notes.append(f"verdict does not pick reference player {pick!r}")
    elif qtype == "trade_eval":
        got = _trade_bucket(verdict_text) or _trade_bucket(full)
        want = _trade_bucket(str(reference.get("pick") or "")) or str(reference.get("pick") or "").upper()
        verdict_ok = got is not None and got == want
        if not verdict_ok:
            notes.append(f"trade verdict {got!r} != reference {want!r}")
    elif qtype == "buy_sell_hold":
        got = _bsh_bucket(verdict_text) or _bsh_bucket(full)
        want = str(reference.get("pick") or "").upper()
        verdict_ok = got is not None and got == want
        if not verdict_ok:
            notes.append(f"buy/sell/hold call {got!r} != reference {want!r}")
    elif qtype == "leaderboard":
        # The reference board IS the verdict: every reference key number must
        # be cited, and the board names must appear in the receipts in the
        # pack's computed order (the model may never reorder a board).
        keys = _canon_key_numbers(reference.get("key_numbers"))
        missing = [k for k in keys if k not in found]
        verdict_ok = not missing
        if missing:
            notes.append(f"key numbers not cited: {', '.join(missing)}")
        board = pack.get("board") or []
        names = [str(r.get("name")) for r in board
                 if isinstance(r, dict) and r.get("name")]
        if names:
            rec_text = " | ".join(
                f"{r.get('label', '')} {r.get('value', '')}"
                for r in (answer.get("receipts") or []) if isinstance(r, dict))
            positions = [rec_text.find(n) for n in names]
            if any(p < 0 for p in positions) or positions != sorted(positions):
                verdict_ok = False
                notes.append("receipt board order does not match the computed board")
    else:  # player, metric, waivers, this_week: key numbers must be cited
        keys = _canon_key_numbers(reference.get("key_numbers"))
        missing = [k for k in keys if k not in found]
        verdict_ok = not missing
        if missing:
            notes.append(f"key numbers not cited: {', '.join(missing)}")

    # must_mention applies on top of every verdict rule
    for needle in expect.get("must_mention") or []:
        if str(needle).lower() not in full.lower():
            verdict_ok = False
            notes.append(f"must_mention missing: {needle!r}")

    # Traps
    trap_ok = True
    if trap.get("kind") == "missing_stat":
        trap_ok = _no_data_shape(answer) and grounding_ok
        if not trap_ok:
            notes.append("missing-stat trap failed: needs the no-data shape and zero invented numbers")
    elif trap.get("kind") == "namesake":
        trap_ok = verdict_ok  # the accept-list check above IS the trap check
    elif trap.get("kind") == "dropped_second_name":
        # The multi-name trap: the question named a second player (or
        # subject) whom a single-subject builder used to drop from the
        # pack entirely, so the answer covered the first name alone and
        # read as complete. The answer must name him: his lite line,
        # his rank line, or the named-gap pointer.
        _dropped = str(trap.get("player") or "").lower()
        trap_ok = bool(_dropped) and _dropped in full.lower() and grounding_ok
        if not trap_ok:
            notes.append("dropped-name trap failed: the second named player "
                         "was not named in the answer")
    elif trap.get("kind") == "pro_gate":
        trap_ok = verdict_ok and grounding_ok
        if not trap_ok:
            notes.append("pro-gate trap failed: needs the gated shape and zero invented numbers")
    elif trap.get("kind") == "no_clean_fit":
        # The trade-finding trap: no partner fits inside the fairness band.
        # The answer must say so plainly (the key-number citation in the
        # verdict check above is what proves the closest miss was named).
        trap_ok = verdict_ok and "no clean fit" in full.lower()
        if not trap_ok:
            notes.append("no-fit trap failed: the answer must say there is no clean fit")
    elif trap.get("kind") == "position_constraint_ignored":
        # The trade-finding position trap: the user constrained the
        # OUTGOING side to one position ("what rb can I trade for X") and
        # the unconstrained best fit is another position's deal (the trap
        # player). In "answered" mode a viable constrained deal exists, so
        # the trap player must not surface anywhere, prose or action rows.
        # In "declined" mode no constrained deal is in range: the answer
        # must decline the position framing, and the trap player may
        # appear only inside the labeled overall fallback, never as the
        # verdict.
        _pos = str(trap.get("position") or "").upper()
        _player = str(trap.get("player") or "").lower()
        _in_actions = any(
            _player and _player in str(a.get("headline") or "").lower()
            for a in (answer or {}).get("actions") or [])
        if trap.get("mode") == "declined":
            trap_ok = bool(_pos) \
                and f"no {_pos.lower()} package" in full.lower() \
                and "closest deal overall" in full.lower() \
                and _player not in verdict_text.lower() \
                and not _in_actions
            if not trap_ok:
                notes.append("position trap failed: the answer must decline the "
                             "position framing and offer the overall fit only "
                             "as a labeled fallback")
        else:
            trap_ok = bool(_player) and _player not in full.lower() and not _in_actions
            if not trap_ok:
                notes.append("position trap failed: the unconstrained best fit "
                             "was presented instead of the constrained deal")
    elif trap.get("kind") == "unneeded_qb_claim":
        # The waiver-gate trap: a value-rising QB sat on the wire, but the
        # roster's QB room was already covered, so the team-fit gate kept
        # him out of the pack. The answer must not recommend claiming him.
        _trap_name = str(trap.get("player") or "").lower()
        _in_actions = any(
            _trap_name and _trap_name in str(a.get("headline") or "").lower()
            for a in (answer or {}).get("actions") or [] if isinstance(a, dict))
        trap_ok = bool(_trap_name) and _trap_name not in full.lower() \
            and not _in_actions and grounding_ok
        if not trap_ok:
            notes.append("unneeded-qb-claim trap failed: the covered QB must not be recommended")
    elif trap.get("kind") == "excluded_player":
        # A screened-out player (a falling knife on a value board) was
        # examined but must not surface anywhere in the answer.
        trap_ok = str(trap.get("name") or "").lower() not in full.lower()
        if not trap_ok:
            notes.append("excluded-player trap failed: the screened-out player was named")
    elif trap.get("kind") == "subset_ranking":
        # The rank-a-named-set trap: the ranking must cover the whole
        # named set in the computed order. A ranking composed from a
        # subset (or in mention order when the computed order differs)
        # fails: every expected name must appear, in order.
        _order = [str(n).lower() for n in trap.get("order") or []]
        _pos = [full.lower().find(n) for n in _order]
        trap_ok = bool(_order) and all(p >= 0 for p in _pos) \
            and _pos == sorted(_pos)
        if not trap_ok:
            notes.append("subset-ranking trap failed: the answer must rank "
                         "the full named set in the computed order")
    elif trap.get("kind") == "unresolved_named":
        # The rank gap trap: one named player could not be resolved.
        # The answer must name him as typed, offer the nearest real
        # match, and rank only the resolved players (their key numbers
        # are enforced by the verdict check above).
        _asked = str(trap.get("asked") or "").lower()
        _near = str(trap.get("nearest") or "").lower()
        trap_ok = bool(_asked) and _asked in full.lower() \
            and (not _near or _near in full.lower())
        if not trap_ok:
            notes.append("unresolved-name trap failed: the answer must name "
                         "the unresolved player and offer the nearest match")
    elif trap.get("kind") == "faller_not_target":
        # The draft-slotting trap: the class headliner ranked above the
        # named pick is presumed taken, so he may surface only as the
        # faller (take him if he somehow drops), never as the target:
        # his name must be in the prose but NOT in the verdict.
        _faller = str(trap.get("name") or "").lower()
        trap_ok = bool(_faller) and _faller not in verdict_text.lower() \
            and _faller in full.lower()
        if not trap_ok:
            notes.append("faller trap failed: the headliner must appear only "
                         "as the faller, never as the target")

    # Structure
    if no_data_expected or clarify or gated_expected:
        structure_ok = bool((answer or {}).get("text"))
        if not structure_ok:
            notes.append("no-data/clarify answer has no text")
    elif qtype == "leaderboard":
        # A board answer carries exactly one receipt per board row; a
        # multi-board pack (defenses vs several positions) counts the
        # rows of every sub-board it carries.
        boards = pack.get("boards") or []
        if len(boards) > 1:
            n_rows = sum(len(b.get("board") or []) for b in boards
                         if isinstance(b, dict))
        else:
            n_rows = len(pack.get("board") or [])
        receipts = (answer or {}).get("receipts") or []
        structure_ok = bool(verdict_text.strip()) and len(receipts) == n_rows
        if not structure_ok:
            notes.append(f"board answer needs a verdict and {n_rows} receipts "
                         f"(one per row), got {len(receipts)}")
    else:
        receipts = (answer or {}).get("receipts") or []
        structure_ok = bool(verdict_text.strip())
        if not structure_ok:
            notes.append("verdict is empty")
        if len(_meaningful_pack_numbers(question)) >= 2:
            if not (2 <= len(receipts) <= 4):
                structure_ok = False
                notes.append(f"receipts count {len(receipts)} outside 2-4 with a number-rich pack")
        elif len(receipts) > 4:
            structure_ok = False
            notes.append(f"receipts count {len(receipts)} over the 4 max")

    # Compare table: asserted only on the cases that expect it (ow22
    # family); the block is code-built, so its check is an exact match
    # against the recomputation above, not a number scan.
    table_ok = True
    if expect.get("compare_table"):
        table_ok, table_notes = _check_compare_table(question, answer)
        notes.extend(table_notes)

    passed = bool(grounding_ok and verdict_ok and trap_ok and structure_ok
                  and table_ok)
    return {"grounding_ok": grounding_ok, "invented": invented,
            "verdict_ok": verdict_ok, "trap_ok": trap_ok,
            "structure_ok": structure_ok, "compare_table_ok": table_ok,
            "passed": passed, "notes": notes}
