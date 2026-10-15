"""Inferred preference proposals + the confirmed risk profile (Stage 2).

Deterministic candidate inference over the account's OWN advice log: no
model call, no cross-account reads. A candidate is a PROPOSAL only; it
becomes a stored preference solely when the user confirms it in the
"What BR remembers about you" panel (the confirmed-inference flow from
the workshop doc). Each candidate carries its evidence: the supporting
log rows, their count, and the most recent one.

Kinds (only the families the doc names, each gated on data that actually
supports it):
- lineup_risk: upside vs floor, from close start/sit calls the user
  backed (thumb up) where the pick was the higher/lower variance option
  by recent-form spread, plus explicit lean words in the user's own
  start/sit questions.
- faab_posture: aggressive vs saver, from waiver answers the user backed
  whose top candidate carried a high/low recommended bid share.
- trade_posture: aggressive vs patient, from how the user thumbed
  code-decided ACCEPT / DECLINE trade verdicts.
- position_lean_<POS>: the position the user asks about most, across
  position-scoped questions.

The risk profile is the confirmed subset of the dimension kinds. It
shapes framing and tie-break language in close calls only; it never
changes a computed number (see compose._PROFILE_SUFFIX).
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics

# kind -> (dimension, value, proposal text). Dimension kinds form the
# risk profile once confirmed.
KIND_INFO = {
    "lineup_risk_upside": ("lineup_risk", "upside",
                           "You usually take the upside in close start/sit calls."),
    "lineup_risk_floor": ("lineup_risk", "floor",
                          "You usually take the safer floor in close start/sit calls."),
    "faab_aggressive": ("faab_posture", "aggressive",
                        "You bid aggressively for waiver targets you want."),
    "faab_saver": ("faab_posture", "saver",
                   "You keep FAAB bids low and hold your budget."),
    "trade_aggressive": ("trade_posture", "aggressive",
                         "You pull the trigger when a trade grades fair or better."),
    "trade_patient": ("trade_posture", "patient",
                      "You hold your players unless a trade clearly wins."),
}

DIMENSION_KINDS = {info[0]: kind for kind, info in KIND_INFO.items()}

# Plain-language profile lines for the memory panel, keyed by
# (dimension, value).
_PROFILE_LINES = {
    ("lineup_risk", "upside"): "Lineup risk: takes the upside in close start/sit calls",
    ("lineup_risk", "floor"): "Lineup risk: takes the safer floor in close start/sit calls",
    ("faab_posture", "aggressive"): "FAAB: bids aggressively for targets they want",
    ("faab_posture", "saver"): "FAAB: keeps bids low and holds budget",
    ("trade_posture", "aggressive"): "Trades: pulls the trigger when a deal grades fair or better",
    ("trade_posture", "patient"): "Trades: holds unless a deal clearly wins",
}

_POSITION_WORDS = [
    ("QB", re.compile(r"\bqb\b|\bquarterbacks?\b")),
    ("RB", re.compile(r"\brb\b|\brunning backs?\b")),
    ("WR", re.compile(r"\bwr\b|\bwide receivers?\b|\breceivers?\b")),
    ("TE", re.compile(r"\bte\b|\btight ends?\b")),
]
_POSITION_PLURAL = {"QB": "quarterbacks", "RB": "running backs",
                    "WR": "wide receivers", "TE": "tight ends"}

_UPSIDE_WORDS = re.compile(r"\bupside|gamble|gambling|ceiling|risky|boom\b")
_FLOOR_WORDS = re.compile(r"\bsafe|safer|safest|floor|steady|conservative\b")

# Evidence thresholds: a leaning becomes a proposal only with repeated,
# one-sided support. Too little evidence means no proposal at all.
_LINEUP_MIN_SIDE = 3
_LINEUP_MIN_SHARE = 0.7
_FAAB_MIN_SIDE = 3
_FAAB_MIN_SHARE = 0.7
_FAAB_AGGRESSIVE_PCT = 15.0
_FAAB_SAVER_PCT = 6.0
_TRADE_MIN_SIGNALS = 4
_TRADE_MIN_SIDE = 3
_TRADE_MIN_SHARE = 0.75
_POSITION_MIN_TOTAL = 5
_POSITION_MIN_TOP = 4
_POSITION_MIN_SHARE = 0.5


def _as_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def evidence_key(kind: str, log_ids) -> str:
    """Fingerprint of one candidate's supporting evidence.

    A dismissed proposal suppresses the same (kind, evidence_key) pair;
    fresh supporting chats change the key, so a proposal can return
    later carrying more evidence rather than the dismissed set.
    """
    ids = ",".join(str(i) for i in sorted(int(i) for i in log_ids if i is not None))
    return hashlib.sha256(f"{kind}|{ids}".encode("utf-8")).hexdigest()


def _candidate(kind, dimension, value, text, signals):
    """signals: list of {log_id, at} on the winning side."""
    ids = [s["log_id"] for s in signals if s.get("log_id") is not None]
    latest = None
    for s in signals:
        at = s.get("at")
        if at is not None and (latest is None or str(at) > str(latest)):
            latest = at
    return {
        "kind": kind, "dimension": dimension, "value": value, "text": text,
        "evidence": {
            "supporting": len(signals),
            "log_ids": ids[:20],
            "latest_at": str(latest) if latest is not None else None,
            "summary": f"From {len(signals)} of your recent chats",
        },
        "evidence_key": evidence_key(kind, ids),
    }


def _winner(signals_by_side: dict, *, min_side: int, min_share: float, min_total: int = 0):
    total = sum(len(v) for v in signals_by_side.values())
    if total < min_total or total == 0:
        return None, []
    side, signals = max(signals_by_side.items(), key=lambda kv: len(kv[1]))
    if len(signals) >= min_side and (len(signals) / total) >= min_share:
        return side, signals
    return None, []


def _form_spread(player: dict):
    form = player.get("recent_form") or []
    points = [f.get("points") for f in form if isinstance(f, dict) and isinstance(f.get("points"), (int, float))]
    if len(points) < 3:
        return None
    return statistics.pstdev([float(p) for p in points])


def _lineup_risk(rows):
    signals = {"upside": [], "floor": []}
    for row in rows:
        if row.get("intent") != "start_sit":
            continue
        entry = {"log_id": row.get("log_id"), "at": row.get("created_at")}
        question = str(row.get("question_text") or "").lower()
        if _UPSIDE_WORDS.search(question):
            signals["upside"].append(entry)
            continue
        if _FLOOR_WORDS.search(question):
            signals["floor"].append(entry)
            continue
        if row.get("thumb") != "up":
            continue
        pack = _as_dict(row.get("evidence_pack"))
        if not ((pack.get("margin") or {}).get("thin")):
            continue
        pick = str(pack.get("pick_player_id") or "")
        spreads = {}
        for p in pack.get("players") or []:
            if isinstance(p, dict) and p.get("player_id") is not None:
                spread = _form_spread(p)
                if spread is not None:
                    spreads[str(p["player_id"])] = spread
        if pick not in spreads or len(spreads) < 2:
            continue
        pick_spread = spreads[pick]
        if pick_spread == max(spreads.values()) and pick_spread > min(spreads.values()):
            signals["upside"].append(entry)
        elif pick_spread == min(spreads.values()) and pick_spread < max(spreads.values()):
            signals["floor"].append(entry)
    side, winning = _winner(signals, min_side=_LINEUP_MIN_SIDE, min_share=_LINEUP_MIN_SHARE)
    if side is None:
        return None
    kind = f"lineup_risk_{side}"
    dimension, value, text = KIND_INFO[kind]
    return _candidate(kind, dimension, value, text, winning)


def _faab_posture(rows):
    signals = {"aggressive": [], "saver": []}
    for row in rows:
        if row.get("intent") != "waivers" or row.get("thumb") != "up":
            continue
        pack = _as_dict(row.get("evidence_pack"))
        if not ((pack.get("faab") or {}).get("enabled")):
            continue
        candidates = pack.get("candidates") or []
        if not candidates or not isinstance(candidates[0], dict):
            continue
        pct = (candidates[0].get("faab") or {}).get("pct_target")
        if not isinstance(pct, (int, float)):
            continue
        entry = {"log_id": row.get("log_id"), "at": row.get("created_at")}
        if float(pct) >= _FAAB_AGGRESSIVE_PCT:
            signals["aggressive"].append(entry)
        elif float(pct) <= _FAAB_SAVER_PCT:
            signals["saver"].append(entry)
    side, winning = _winner(signals, min_side=_FAAB_MIN_SIDE, min_share=_FAAB_MIN_SHARE)
    if side is None:
        return None
    kind = f"faab_{side}"
    dimension, value, text = KIND_INFO[kind]
    return _candidate(kind, dimension, value, text, winning)


def _trade_posture(rows):
    signals = {"aggressive": [], "patient": []}
    for row in rows:
        if row.get("intent") != "trade_eval" or row.get("thumb") not in ("up", "down"):
            continue
        pack = _as_dict(row.get("evidence_pack"))
        call = pack.get("trade_verdict")
        if call not in ("ACCEPT", "DECLINE"):
            verdict = _as_dict(row.get("verdict"))
            call = verdict.get("call")
        if call not in ("ACCEPT", "DECLINE"):
            continue
        entry = {"log_id": row.get("log_id"), "at": row.get("created_at")}
        if (call == "ACCEPT") == (row.get("thumb") == "up"):
            signals["aggressive"].append(entry)
        else:
            signals["patient"].append(entry)
    side, winning = _winner(signals, min_side=_TRADE_MIN_SIDE,
                            min_share=_TRADE_MIN_SHARE, min_total=_TRADE_MIN_SIGNALS)
    if side is None:
        return None
    kind = f"trade_{side}"
    dimension, value, text = KIND_INFO[kind]
    return _candidate(kind, dimension, value, text, winning)


def _position_lean(rows):
    counts: dict[str, list] = {}
    total = 0
    for row in rows:
        if row.get("intent") not in ("start_sit", "waivers", "metric", "buy_sell_hold", "player"):
            continue
        question = str(row.get("question_text") or "").lower()
        for pos, pattern in _POSITION_WORDS:
            if pattern.search(question):
                counts.setdefault(pos, []).append(
                    {"log_id": row.get("log_id"), "at": row.get("created_at")})
                total += 1
                break
    if total < _POSITION_MIN_TOTAL or not counts:
        return None
    pos, signals = max(counts.items(), key=lambda kv: len(kv[1]))
    if len(signals) < _POSITION_MIN_TOP or (len(signals) / total) < _POSITION_MIN_SHARE:
        return None
    kind = f"position_lean_{pos.lower()}"
    text = f"You ask about {_POSITION_PLURAL[pos]} more than any other position."
    cand = _candidate(kind, None, pos, text, signals)
    cand["evidence"]["summary"] = (
        f"From {len(signals)} of your {total} position questions")
    return cand


def candidates_from_log(rows) -> list[dict]:
    """All current proposals supported by these log rows (the account's
    own rows only; callers fetch them keyed by account). Deterministic:
    same rows in, same candidates out."""
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    out = []
    for fn in (_lineup_risk, _faab_posture, _trade_posture, _position_lean):
        try:
            cand = fn(rows)
        except Exception:
            cand = None
        if cand:
            out.append(cand)
    return out


def profile_from_preferences(preferences) -> dict:
    """The confirmed risk profile: dimension -> value, from stored
    preferences that came from confirmed inferences (source='inferred'
    with a dimension kind). Explicit preferences never set a dimension."""
    profile: dict[str, str] = {}
    for pref in preferences or []:
        if not isinstance(pref, dict) or pref.get("source") != "inferred":
            continue
        info = KIND_INFO.get(str(pref.get("kind") or ""))
        if info:
            profile[info[0]] = info[1]
    return profile


def profile_lines(profile: dict) -> list[str]:
    """Plain-language profile lines for the memory panel."""
    lines = []
    for (dimension, value), line in _PROFILE_LINES.items():
        if (profile or {}).get(dimension) == value:
            lines.append(line)
    return lines
