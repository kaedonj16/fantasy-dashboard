"""The weekly review packet (workshop sections 3 and 6, stage 2).

Auto-generated before Kaedon's weekly review ritual; she only reads and
decides. The packet is hers alone: the generator writes a Markdown file,
and the admin analytics Assistant section carries a compact summary
block. Nothing here is user-facing, and nothing here self-modifies: the
packet reports grades, misses, and correction clusters; proposed changes
and data tickets stay proposals until she approves them in the ritual.

Contents (section 3): volume and thumbs by question type; graded hit
rate by type for the cycle with the trend vs the prior cycle; the worst
misses with question, call, outcome, and receipts side by side; every
downvoted answer with its outcome grade next to the thumb; correction
clusters (3 or more disputes on the same player and claim) pre-written
as draft data tickets, plus the rollup by correction kind; unanswered
counts; the proactive report; and grading-debt flags for any graded type
whose grader has gone quiet while due rows wait.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

from dashboard_services.assistant import store

logger = logging.getLogger(__name__)

CYCLE_DAYS = 7
CLUSTER_MIN = 3  # the doc's cluster threshold for a draft data ticket
GRADED_TYPES = ("start_sit", "buy_sell_hold")

_CATEGORY_LABELS = {
    "wrong_data": "Wrong data",
    "bad_reasoning": "Bad reasoning",
    "ignored_situation": "Ignored situation",
    "other": "Other",
}


def _hit_rate(rows) -> float | None:
    graded = [r for r in rows if r.get("grade") in ("hit", "partial", "miss")]
    if not graded:
        return None
    hits = sum(1 for r in graded if r["grade"] == "hit")
    return round(hits / len(graded), 3)


def _grade_counts(rows) -> dict:
    out = {"graded": 0, "hit": 0, "partial": 0, "miss": 0}
    for r in rows:
        g = r.get("grade")
        if g in ("hit", "partial", "miss"):
            out["graded"] += 1
            out[g] += 1
    return out


def _call_summary(row) -> str:
    verdict = row.get("verdict")
    if isinstance(verdict, dict) and verdict.get("call"):
        return str(verdict["call"])
    detail = row.get("outcome_detail") or {}
    if row.get("intent") == "start_sit" and detail.get("pick"):
        return f"Start {detail['pick'].get('name')}"
    if row.get("intent") == "buy_sell_hold" and detail.get("call"):
        return f"{str(detail['call']).upper()} {detail.get('name') or ''}".strip()
    return "(no recorded call)"


def _outcome_summary(row) -> str:
    detail = row.get("outcome_detail") or {}
    if row.get("intent") == "start_sit" and detail.get("pick"):
        alt = detail.get("best_alternative") or {}
        return (f"{detail['pick'].get('name')} scored {detail['pick'].get('points')} vs "
                f"{alt.get('name')} {alt.get('points')} (delta {detail.get('delta')})")
    if row.get("intent") == "buy_sell_hold" and "value_delta_pct" in detail:
        return (f"{detail.get('name')} value {detail.get('value_at_answer')} on "
                f"{detail.get('value_at_answer_date')} to {detail.get('value_now')} on "
                f"{detail.get('value_now_date')} ({detail.get('value_delta_pct')} percent)")
    return "(no recorded outcome detail)"


def _miss_severity(row) -> float:
    detail = row.get("outcome_detail") or {}
    if row.get("intent") == "start_sit":
        try:
            return -float(detail.get("delta") or 0)
        except (TypeError, ValueError):
            return 0.0
    if row.get("intent") == "buy_sell_hold":
        try:
            pct = float(detail.get("value_delta_pct") or 0)
            call = detail.get("call")
            return pct if call == "sell" else -pct
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _player_key(row) -> str:
    pack = row.get("evidence_pack") if isinstance(row.get("evidence_pack"), dict) else {}
    player = pack.get("player") or {}
    if isinstance(player, dict) and player.get("name"):
        return str(player["name"])
    players = pack.get("players") or []
    pick_id = str(pack.get("pick_player_id") or "")
    for p in players:
        if isinstance(p, dict) and str(p.get("player_id")) == pick_id and p.get("name"):
            return str(p["name"])
    if players and isinstance(players[0], dict) and players[0].get("name"):
        return str(players[0]["name"])
    entities = row.get("entities") if isinstance(row.get("entities"), dict) else {}
    names = entities.get("player_names") or []
    return str(names[0]) if names else "unknown player"


def build_packet(current_rows, graded_rows, prior_graded_rows, *, generated_at,
                 window_start, window_end, ungraded_backlog=None) -> dict:
    """Assemble the packet dict from already-fetched log rows.

    current_rows: rows created in the window (volume, thumbs, corrections).
    graded_rows: rows whose grade was written in the window (hit rates,
    worst misses). prior_graded_rows: the same for the prior window
    (trend). ungraded_backlog: {intent: count} of due-but-ungraded rows,
    for the grading-debt flags.
    """
    volume: dict = {}
    for r in current_rows:
        intent = r.get("intent") or "unknown"
        v = volume.setdefault(intent, {"count": 0, "up": 0, "down": 0})
        v["count"] += 1
        if r.get("thumb") == "up":
            v["up"] += 1
        elif r.get("thumb") == "down":
            v["down"] += 1

    graded_by_type: dict = {}
    for r in graded_rows:
        intent = r.get("intent") or "unknown"
        graded_by_type.setdefault(intent, []).append(r)
    prior_by_type: dict = {}
    for r in prior_graded_rows:
        intent = r.get("intent") or "unknown"
        prior_by_type.setdefault(intent, []).append(r)
    graded_section = {}
    for intent in sorted(set(graded_by_type) | set(prior_by_type)):
        cur = _grade_counts(graded_by_type.get(intent, []))
        cur_rate = _hit_rate(graded_by_type.get(intent, []))
        prior_rate = _hit_rate(prior_by_type.get(intent, []))
        graded_section[intent] = {
            **cur,
            "hit_rate": cur_rate,
            "prior_hit_rate": prior_rate,
            "trend": (round(cur_rate - prior_rate, 3)
                      if cur_rate is not None and prior_rate is not None else None),
        }

    misses = [r for r in graded_rows if r.get("grade") == "miss"]
    misses.sort(key=_miss_severity, reverse=True)
    worst_misses = [
        {
            "log_id": r.get("log_id"), "intent": r.get("intent"),
            "question": r.get("question_text") or "",
            "call": _call_summary(r),
            "outcome": _outcome_summary(r),
            "receipts": r.get("numbers_shown") or {},
            "thumb": r.get("thumb"),
        }
        for r in misses[:10]
    ]

    downvoted = [
        {
            "log_id": r.get("log_id"), "intent": r.get("intent"),
            "question": r.get("question_text") or "",
            "grade": r.get("grade") or "ungraded",
        }
        for r in current_rows if r.get("thumb") == "down"
    ]

    corrections = [r for r in current_rows if r.get("correction_category")]
    by_kind: dict = {}
    for r in corrections:
        kind = r.get("correction_category") or "other"
        by_kind[kind] = by_kind.get(kind, 0) + 1
    clusters: dict = {}
    for r in corrections:
        claim = r.get("corrected_claim") or r.get("correction_category") or "unspecified claim"
        key = (_player_key(r), str(claim))
        c = clusters.setdefault(key, {"count": 0, "category": r.get("correction_category"),
                                      "samples": []})
        c["count"] += 1
        if r.get("correction_text") and len(c["samples"]) < 3:
            c["samples"].append(str(r["correction_text"]))
    draft_tickets = [
        {"player": player, "claim": claim, "count": c["count"],
         "category": c["category"], "samples": c["samples"]}
        for (player, claim), c in sorted(clusters.items(), key=lambda kv: -kv[1]["count"])
        if c["count"] >= CLUSTER_MIN
    ]

    unanswered = {
        "out_of_scope": sum(1 for r in current_rows if r.get("intent") == "out_of_scope"),
        "no_data_by_intent": {},
    }
    for r in current_rows:
        pack = r.get("evidence_pack") if isinstance(r.get("evidence_pack"), dict) else {}
        if pack.get("available") is False:
            intent = r.get("intent") or "unknown"
            unanswered["no_data_by_intent"][intent] = (
                unanswered["no_data_by_intent"].get(intent, 0) + 1)

    proactive_rows = [r for r in current_rows if r.get("proactive")]
    proactive = {"total": len(proactive_rows), "by_trigger": {}}
    for r in proactive_rows:
        t = r.get("trigger") or "unknown"
        proactive["by_trigger"][t] = proactive["by_trigger"].get(t, 0) + 1

    debt_flags = []
    backlog = ungraded_backlog or {}
    for intent in GRADED_TYPES:
        graded_this_cycle = graded_section.get(intent, {}).get("graded", 0)
        if backlog.get(intent, 0) > 0 and graded_this_cycle == 0:
            debt_flags.append(
                f"{intent}: {backlog[intent]} due rows are waiting and the grader "
                "wrote no grades this cycle (grading debt)")

    return {
        "generated_at": generated_at.isoformat() if hasattr(generated_at, "isoformat") else str(generated_at),
        "window_start": window_start.isoformat() if hasattr(window_start, "isoformat") else str(window_start),
        "window_end": window_end.isoformat() if hasattr(window_end, "isoformat") else str(window_end),
        "volume_by_type": volume,
        "graded_by_type": graded_section,
        "worst_misses": worst_misses,
        "downvoted": downvoted,
        "corrections_by_kind": by_kind,
        "draft_tickets": draft_tickets,
        "unanswered": unanswered,
        "proactive": proactive,
        "grading_debt_flags": debt_flags,
    }


def _fmt_rate(rate) -> str:
    return "n/a" if rate is None else f"{rate:.0%}"


def render_markdown(packet: dict) -> str:
    lines = [
        "# BR Assistant weekly review packet",
        "",
        f"Window: {packet['window_start']} to {packet['window_end']}",
        f"Generated: {packet['generated_at']}",
        "",
        "## Volume and thumbs by question type",
        "",
    ]
    if packet["volume_by_type"]:
        for intent, v in sorted(packet["volume_by_type"].items(),
                                key=lambda kv: -kv[1]["count"]):
            lines.append(f"- {intent}: {v['count']} answers, {v['up']} thumbs up, {v['down']} thumbs down")
    else:
        lines.append("- No answers logged this cycle.")

    lines += ["", "## Graded hit rate by type", ""]
    if packet["graded_by_type"]:
        for intent, g in sorted(packet["graded_by_type"].items()):
            trend = ""
            if g.get("trend") is not None:
                trend = f", trend {g['trend']:+.0%} vs prior cycle"
            lines.append(
                f"- {intent}: {g['graded']} graded, {g['hit']} hit, {g['partial']} partial, "
                f"{g['miss']} miss, hit rate {_fmt_rate(g['hit_rate'])} "
                f"(prior {_fmt_rate(g['prior_hit_rate'])}){trend}")
    else:
        lines.append("- No grades written this cycle.")

    lines += ["", "## Worst misses", ""]
    if packet["worst_misses"]:
        for m in packet["worst_misses"]:
            lines.append(f"- [{m['intent']}] Q: {m['question']}")
            lines.append(f"  Call: {m['call']}")
            lines.append(f"  Outcome: {m['outcome']}")
            if m["receipts"]:
                receipts = ", ".join(f"{k}: {v}" for k, v in m["receipts"].items())
                lines.append(f"  Receipts: {receipts}")
    else:
        lines.append("- No misses graded this cycle.")

    lines += ["", "## Downvoted answers (thumb vs outcome grade)", ""]
    if packet["downvoted"]:
        for d in packet["downvoted"]:
            lines.append(f"- [{d['intent']}] grade: {d['grade']} | Q: {d['question']}")
    else:
        lines.append("- No downvotes this cycle.")

    lines += ["", "## Corrections", ""]
    if packet["corrections_by_kind"]:
        for kind, n in sorted(packet["corrections_by_kind"].items(), key=lambda kv: -kv[1]):
            lines.append(f"- {_CATEGORY_LABELS.get(kind, kind)}: {n}")
    else:
        lines.append("- No corrections filed this cycle.")
    if packet["draft_tickets"]:
        lines += ["", "### Draft data tickets (clusters of 3 or more)", ""]
        for t in packet["draft_tickets"]:
            lines.append(
                f"- {t['count']} disputes about {t['player']}: claim \"{t['claim']}\" "
                f"(kind: {_CATEGORY_LABELS.get(t['category'], t['category'])})")
            for s in t["samples"]:
                lines.append(f"  User said: {s}")
    else:
        lines += ["", "No correction cluster reached the ticket threshold this cycle."]

    lines += ["", "## Unanswered and no-data", ""]
    lines.append(f"- Out of scope questions: {packet['unanswered']['out_of_scope']}")
    if packet["unanswered"]["no_data_by_intent"]:
        for intent, n in sorted(packet["unanswered"]["no_data_by_intent"].items()):
            lines.append(f"- No-data answers ({intent}): {n}")

    lines += ["", "## Proactive", ""]
    lines.append(f"- Proactive answers sent: {packet['proactive']['total']}")
    for t, n in sorted(packet["proactive"]["by_trigger"].items()):
        lines.append(f"- {t}: {n}")

    lines += ["", "## Grading debt", ""]
    if packet["grading_debt_flags"]:
        for f in packet["grading_debt_flags"]:
            lines.append(f"- FLAG: {f}")
    else:
        lines.append("- No grading debt flags.")

    lines.append("")
    return "\n".join(lines)


def generate_packet(*, now=None, out_dir=None) -> tuple[str, dict]:
    """Fetch the last two cycles and write the Markdown packet file.

    Returns (path, packet). The output directory defaults to the
    ASSISTANT_PACKET_DIR env var, else ./assistant_review_packets.
    """
    now = now or datetime.now(timezone.utc)
    window_end = now
    window_start = now - timedelta(days=CYCLE_DAYS)
    prior_start = now - timedelta(days=2 * CYCLE_DAYS)
    current_rows = store.fetch_log_rows_between(window_start, window_end)
    graded_rows = store.fetch_graded_rows_between(window_start, window_end)
    prior_graded_rows = store.fetch_graded_rows_between(prior_start, window_start)
    backlog = {intent: len(store.fetch_ungraded_rows(intent, limit=1000))
               for intent in GRADED_TYPES}
    packet = build_packet(current_rows, graded_rows, prior_graded_rows,
                          generated_at=now, window_start=window_start,
                          window_end=window_end, ungraded_backlog=backlog)
    out_dir = out_dir or os.environ.get("ASSISTANT_PACKET_DIR") or "assistant_review_packets"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"assistant_review_packet_{now.date().isoformat()}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(packet))
    return path, packet
