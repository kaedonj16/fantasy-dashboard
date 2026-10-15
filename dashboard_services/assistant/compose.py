"""Answer composition: pack in, answer out. This is the eval seam.

Pure with respect to tools: it only calls the model (via model.call_model_json,
monkeypatched in tests) and post-checks grounding in code. Grounding is
enforced on EVERY surface, not just receipts: numbers in the verdict, prose,
caveat, and follow-ups must all appear in the evidence pack (or the question
itself). A draft that states an unverifiable number is regenerated once with
a stricter instruction; if the retry still fails, the offending sentences and
numbers are stripped rather than shipped. Receipts whose numbers do not all
appear in the pack JSON are dropped.

The answer dict is the UI contract (static/app.js initAssistant): act_link is
an object {href, label} (or None) assigned deterministically in code, never
by the model; this-week actions carry href, cta_label, and the pack's
key_number so the UI can render them as Next Steps rows.
"""
from __future__ import annotations

import json
import logging
import re

from dashboard_services.assistant import PROMPT_VERSION
from dashboard_services.assistant.model import call_model_json

logger = logging.getLogger(__name__)

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": ["string", "null"]},
        "text": {"type": "string"},
        "receipts": {"type": "array", "items": {"type": "object", "properties": {
            "label": {"type": "string"}, "value": {"type": "string"}}, "required": ["label", "value"], "additionalProperties": False}},
        "caveat": {"type": ["string", "null"]},
        "followups": {"type": "array", "items": {"type": "string"}},
        "act_link": {"type": ["string", "null"]},
    },
    "required": ["verdict", "text", "receipts", "caveat", "followups", "act_link"],
    "additionalProperties": False,
}

SYSTEM = (
    "You are the BR Fantasy assistant. Answer from the evidence pack ONLY. "
    "Verdict first, in the verdict field. Every number you state must appear verbatim in the pack; "
    "never use a number from memory. Receipts: 2 to 4 entries, each a label and the pack value as a string. "
    "Caveat only when the pack marks margin.thin true or data stale, otherwise null. "
    "If assumed_context.assumed_league_uncertain is true, state the assumed league in the text. "
    "If a stored preference shapes the answer, name that preference in the text. "
    "No em dashes. Use plain hyphens. Short, direct sentences. Return JSON only."
)

# Deterministic act links per intent (UI contract: {href, label} or None).
_ACT_LINKS = {
    "start_sit": {"href": "/waivers?tab=startsit", "label": "Open Start/Sit"},
    "trade_eval": {"href": "/trade-hub", "label": "Open in Trade Hub"},
    "buy_sell_hold": {"href": "/players", "label": "View player"},
    "player": {"href": "/players", "label": "View player"},
    "metric": {"href": "/advanced-metrics", "label": "Open Advanced Metrics"},
    "waivers": {"href": "/waivers", "label": "Open Waivers"},
    "this_week": {"href": "/", "label": "Open dashboard"},
}

# CTA labels for this-week action rows, keyed by the pack action's tag; the
# digest vocabulary (lineup / waiver / trade) matches the site's own rows.
_CTA_LABELS = {"lineup": "Set lineup", "waiver": "View waivers", "waivers": "View waivers",
               "trade": "Build offer"}


def _cta_label(action: dict) -> str:
    if action.get("cta_label"):
        return str(action["cta_label"])
    tag = str(action.get("tag") or "").lower()
    for key, label in _CTA_LABELS.items():
        if key in tag:
            return label
    return "View"


# Standalone numeric token (same discipline as the eval scorer): digits glued
# to letters or codes (WR1, W7, 2026season) are labels, not numeric claims.
_NUM_RE = re.compile(r"(?<![A-Za-z0-9])\$?-?\d[\d,]*(?:\.\d+)?%?(?![A-Za-z0-9])")


def _numbers_in(text: str) -> list[str]:
    return _NUM_RE.findall(text or "")


def _norm_num(token: str) -> str:
    return str(token).replace("$", "").replace(",", "").rstrip("%").strip()


def _pack_numbers(pack) -> set[str]:
    raw = json.dumps(pack or {}, default=str)
    found = set(_numbers_in(raw))
    # Also accept integer forms of float pack values (12.0 in pack, "12" in receipt).
    extra = set()
    for n in found:
        try:
            f = float(n.rstrip("%"))
            if f == int(f):
                extra.add(str(int(f)))
                extra.add(f"{int(f)}.0")
        except Exception:
            pass
    return found | extra


def _ground_set(pack, question: str | None = None) -> set[str]:
    """Normalized numbers an answer may state: the pack's, plus the question's.

    Normalized (commas and % stripped) with sign variants, so "17.9%" in prose
    grounds against 17.9 in the pack and a pack drop of -20.1 can be stated as
    a 20.1 magnitude. A differently-rounded value (17.9 stated as 18) does NOT
    ground; that is an altered number, not a formatting choice.
    """
    allowed: set[str] = set()
    for token in _pack_numbers(pack):
        norm = _norm_num(token)
        if norm:
            allowed.add(norm)
            allowed.add(norm.lstrip("-"))
            allowed.add("-" + norm.lstrip("-"))
    for token in _numbers_in(question or ""):
        norm = _norm_num(token)
        if norm:
            allowed.add(norm)
            allowed.add(norm.lstrip("-"))
    return allowed


def _receipt_grounded(receipt_value: str, allowed: set[str]) -> bool:
    return all(_norm_num(n) in allowed for n in _numbers_in(receipt_value))


def _ungrounded_numbers(data: dict, allowed: set[str]) -> list[str]:
    """Numbers stated in verdict/prose/caveat/follow-ups that are not allowed."""
    texts = [data.get("verdict") or "", data.get("text") or "", data.get("caveat") or ""]
    texts += [str(f) for f in (data.get("followups") or [])]
    bad: list[str] = []
    for text in texts:
        for token in _numbers_in(str(text)):
            if _norm_num(token) not in allowed and token not in bad:
                bad.append(token)
    return bad


def _strip_ungrounded(data: dict, allowed: set[str]) -> dict:
    """Last-resort cleanup: remove what cannot be verified, keep the rest.

    Prose loses the offending sentences; the verdict and caveat lose the
    offending number tokens; follow-ups carrying one are dropped whole.
    """
    out = dict(data)
    text = str(out.get("text") or "")
    if text:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        kept = [s for s in sentences
                if all(_norm_num(n) in allowed for n in _numbers_in(s))]
        out["text"] = " ".join(kept).strip()
        if not out["text"]:
            out["text"] = "See the receipts below for the numbers behind this call."
    for field in ("verdict", "caveat"):
        value = out.get(field)
        if value:
            cleaned = str(value)
            for token in _numbers_in(cleaned):
                if _norm_num(token) not in allowed:
                    cleaned = cleaned.replace(token, "")
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
            out[field] = cleaned or None
    out["followups"] = [
        f for f in (out.get("followups") or [])
        if all(_norm_num(n) in allowed for n in _numbers_in(str(f)))
    ]
    return out


def no_data_answer(pack: dict) -> dict:
    missing = pack.get("missing") or "that data is not tracked"
    nearest = pack.get("nearest") or []
    text = f"That is not available: {missing}."
    if nearest:
        text += f" The nearest tracked options are {', '.join(str(n) for n in nearest[:3])}. Ask about one of those and I can pull the live numbers."
    return {"verdict": None, "text": text, "receipts": [], "caveat": None,
            "followups": [f"What is {n}?" for n in nearest[:2]],
            "act_link": dict(_ACT_LINKS["metric"]), "actions": []}


_STRICT_SUFFIX = (
    " STRICT MODE: your previous draft stated numbers that are NOT in the evidence pack. "
    "Use only numbers that appear verbatim in the evidence pack or in the question. "
    "If a number you want is not in the pack, leave the number out entirely."
)


def compose_answer(intent: str, question: str, pack: dict, *, preferences=None, assumed_context=None, history=None) -> dict:
    """Compose an answer dict from an evidence pack. Pure eval seam."""
    if isinstance(pack, dict) and pack.get("available") is False:
        return no_data_answer(pack)
    payload = {
        "intent": intent,
        "question": question,
        "evidence_pack": pack,
        "preferences": [p.get("text") if isinstance(p, dict) else str(p) for p in (preferences or [])],
        "assumed_context": assumed_context or {},
        "history": (history or [])[-3:],
        "prompt_version": PROMPT_VERSION,
    }
    allowed = _ground_set(pack, question)
    data = call_model_json(SYSTEM, json.dumps(payload, default=str), SCHEMA, "assistant_answer")
    bad = _ungrounded_numbers(data, allowed)
    if bad:
        logger.info("[assistant] ungrounded prose numbers, regenerating once: %s", bad)
        retry_payload = dict(payload)
        retry_payload["grounding_retry"] = {
            "banned_numbers": bad,
            "instruction": "Your previous draft used numbers that are not in the evidence pack. "
                           "Do not use them. Restate the answer using only pack numbers.",
        }
        try:
            retry = call_model_json(SYSTEM + _STRICT_SUFFIX, json.dumps(retry_payload, default=str),
                                    SCHEMA, "assistant_answer")
        except Exception:
            retry = None
        if retry is not None and not _ungrounded_numbers(retry, allowed):
            data = retry
        else:
            if retry is not None:
                data = retry
            data = _strip_ungrounded(data, allowed)
            logger.info("[assistant] stripped ungrounded prose numbers: %s",
                        _ungrounded_numbers(data, allowed) or bad)
    receipts = []
    for r in (data.get("receipts") or [])[:4]:
        if not isinstance(r, dict):
            continue
        label = str(r.get("label") or "")
        value = str(r.get("value") or "")
        if _receipt_grounded(value, allowed):
            receipts.append({"label": label, "value": value})
        else:
            logger.info("[assistant] dropped ungrounded receipt %s=%s", label, value)
    caveat = data.get("caveat")
    margin_thin = bool(((pack or {}).get("margin") or {}).get("thin"))
    if caveat and not margin_thin and not (pack or {}).get("stale"):
        caveat = None
    actions = []
    if intent == "this_week":
        default_href = (_ACT_LINKS.get("this_week") or {}).get("href")
        for a in (pack or {}).get("actions") or []:
            if isinstance(a, dict):
                actions.append({
                    "tag": a.get("tag"), "headline": a.get("headline"),
                    "detail": a.get("detail"),
                    "href": a.get("act_link") or a.get("href") or default_href,
                    "cta_label": _cta_label(a),
                    "key_number": a.get("key_number"),
                })
    act_link = _ACT_LINKS.get(intent)
    return {
        "verdict": data.get("verdict"),
        "text": data.get("text") or "",
        "receipts": receipts,
        "caveat": caveat,
        "followups": (data.get("followups") or [])[:3],
        "act_link": dict(act_link) if act_link else None,
        "actions": actions,
    }


def numbers_shown_from(receipts) -> dict:
    return {r["label"]: r["value"] for r in (receipts or []) if isinstance(r, dict) and r.get("label")}
