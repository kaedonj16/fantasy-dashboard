from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class IdentityMatch:
    player_id: str
    confidence: float
    method: str
    ambiguous: bool = False
    candidates: Optional[List[str]] = None


def _norm_name(value: str) -> str:
    # Strip periods before main regex so "K.C." and "KC" both normalize to "kc"
    s = (value or "").lower().replace(".", "")
    base = re.sub(r"[^a-z0-9]+", " ", s).strip()
    normalized = " ".join(base.split())
    # Collapse spaced single-letter initials: "k c concepcion" -> "kc concepcion"
    normalized = re.sub(r"\b([a-z])\s([a-z])\b", r"\1\2", normalized)
    return normalized


def _norm_school(value: str) -> str:
    return _norm_name(value)


def build_identity_index(prospects: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    index: Dict[str, Dict[str, List[str]]] = {
        "player_id": {},
        "sleeper_id": {},
        "college_id": {},
        "cfb_id": {},
        "fallback": {},
    }

    for p in prospects:
        pid = p.get("player_id")
        if not pid:
            continue
        for key, field in (("sleeper_id", "sleeper_id"), ("college_id", "college_player_id"), ("cfb_id", "cfb_id")):
            value = p.get(field)
            if value:
                index[key].setdefault(str(value), []).append(pid)

        index["player_id"].setdefault(str(pid), []).append(pid)
        fallback = _fallback_key(
            p.get("name", ""),
            p.get("school", ""),
            p.get("position", ""),
            p.get("draft_class_year") or p.get("season") or 0,
        )
        index["fallback"].setdefault(fallback, []).append(pid)

    return index


def _fallback_key(name: str, school: str, position: str, season: int) -> str:
    return f"{_norm_name(name)}|{_norm_school(school)}|{(position or '').upper()}|{season}"


def reconcile_player_identity(
    record: Dict[str, Any],
    identity_index: Dict[str, Dict[str, List[str]]],
) -> Optional[IdentityMatch]:
    direct_keys: List[Tuple[str, str]] = [
        ("player_id", "player_id"),
        ("sleeper_id", "sleeper_id"),
        ("college_player_id", "college_id"),
        ("cfb_id", "cfb_id"),
    ]

    for record_field, index_bucket in direct_keys:
        value = record.get(record_field)
        if not value:
            continue
        candidates = identity_index[index_bucket].get(str(value), [])
        if len(candidates) == 1:
            return IdentityMatch(player_id=candidates[0], confidence=0.99, method=f"direct:{record_field}")
        if len(candidates) > 1:
            return IdentityMatch(
                player_id=candidates[0],
                confidence=0.0,
                method=f"ambiguous:{record_field}",
                ambiguous=True,
                candidates=sorted(set(candidates)),
            )

    fallback = _fallback_key(
        record.get("name", ""),
        record.get("school", ""),
        record.get("position", ""),
        int(record.get("draft_class_year") or record.get("season") or 0),
    )
    candidates = identity_index["fallback"].get(fallback, [])
    if len(candidates) == 1:
        return IdentityMatch(player_id=candidates[0], confidence=0.85, method="fallback:name_school_position_season")
    if len(candidates) > 1:
        return IdentityMatch(
            player_id=candidates[0],
            confidence=0.0,
            method="ambiguous:fallback",
            ambiguous=True,
            candidates=sorted(set(candidates)),
        )
    return None
