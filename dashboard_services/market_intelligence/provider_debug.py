"""Bounded, opt-in diagnostics for SportsGameOdds response schema changes."""
from __future__ import annotations

import json
import os
from pathlib import Path

DEBUG_ENV = "MARKET_DEBUG_PROVIDER_RESPONSES"
MAX_EVENTS = 3
MAX_PLAYER_ODDS = 5
MAX_TEAM_ODDS = 10
MAX_BOOK_ROWS = 3
MAX_TEAM_FAILURES = 5
MAX_UNKNOWN_TOKENS = 10
SNAPSHOT_PATH = Path("/tmp/sportsgameodds_debug_sample.json")

_SKIP_KEYS = {"odds", "players", "markets", "books", "bybookmaker"}
_SENSITIVE_PARTS = ("apikey", "api_key", "authorization", "credential", "header", "secret",
                    "token", "cookie", "password")
_IDENTITY_PARTS = ("team", "home", "away", "participant", "competitor", "alignment",
                   "side", "name", "abbreviation", "shortname")
_ODD_FIELDS = ("oddID", "statID", "statEntityID", "periodID", "betTypeID", "sideID",
               "playerID", "teamID", "participantID", "marketName", "bookOverUnder")
_BOOK_FIELDS = ("available", "odds", "overUnder", "spread", "lastUpdated", "updatedAt",
                "status", "suspended")
_PLAYER_STAT_PARTS = ("passing", "rushing", "receiving", "reception", "touchdown",
                      "interception")


def debug_enabled() -> bool:
    return os.getenv(DEBUG_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _safe_scalar(value):
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:200]


def sanitize_identity(value, depth: int = 0):
    """Retain shallow identity fields while excluding odds and sensitive data."""
    if depth > 3:
        return "<max-depth>"
    if isinstance(value, dict):
        out = {}
        for key, child in list(value.items())[:50]:
            low = str(key).lower()
            if low in _SKIP_KEYS or any(part in low for part in _SENSITIVE_PARTS):
                continue
            if depth <= 1 or any(part in low for part in _IDENTITY_PARTS):
                out[str(key)] = sanitize_identity(child, depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [sanitize_identity(child, depth + 1) for child in value[:3]]
    return _safe_scalar(value)


def sanitize_odd(odd: dict) -> dict:
    out = {key: _safe_scalar(odd.get(key)) for key in _ODD_FIELDS}
    nested = odd.get("byBookmaker")
    out["byBookmaker"] = list(nested)[:20] if isinstance(nested, dict) else []
    return out


def _json(value) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, default=str)


class SportsGameOddsDebug:
    """One-client debug session with hard sample and recursion bounds."""

    def __init__(self, enabled: bool | None = None, snapshot_path: Path | None = None):
        self.enabled = debug_enabled() if enabled is None else enabled
        self.snapshot_path = snapshot_path or SNAPSHOT_PATH
        self.pages = self.events = self.player_odds = self.team_odds = self.book_rows = 0
        self.team_failures = 0
        self.unknown_tokens: set[str] = set()
        self.snapshot_events = []

    def response_page(self, payload: dict) -> None:
        if not self.enabled or self.pages:
            return
        self.pages += 1
        data = payload.get("data")
        cursor = payload.get("nextCursor") or (
            payload.get("meta", {}).get("nextCursor") if isinstance(payload.get("meta"), dict) else None)
        print(f"[market-debug] SportsGameOdds response keys: {', '.join(sorted(map(str, payload)))}")
        print(f"[market-debug] data type: {type(data).__name__}")
        print(f"[market-debug] data count: {len(data) if isinstance(data, list) else 0}")
        print(f"[market-debug] next cursor present: {str(bool(cursor)).lower()}")

    def event(self, event: dict) -> None:
        if not self.enabled:
            return
        sample_event = self.events < MAX_EVENTS
        if sample_event:
            self.events += 1
            print(f"[market-debug] event {self.events} keys: {', '.join(sorted(map(str, event)))}")
            print(f"[market-debug] eventID: {_safe_scalar(event.get('eventID') or event.get('id'))}")
            print(f"[market-debug] event type: {_safe_scalar(event.get('eventType') or event.get('type'))}")
            print(f"[market-debug] leagueID: {_safe_scalar(event.get('leagueID'))}")
            identity = {key: event[key] for key in event if any(
                part in str(key).lower() for part in _IDENTITY_PARTS) and str(key).lower() not in _SKIP_KEYS}
            safe_identity = sanitize_identity(identity)
            print(f"[market-debug] sample event identity={_json(safe_identity)}")
            for key in ("teams", "participants", "competitors", "teamsById"):
                value = event.get(key)
                if value is None:
                    continue
                shape = ({"type": "dict", "keys": list(value)[:10]} if isinstance(value, dict) else
                         {"type": "list", "count": len(value)} if isinstance(value, list) else
                         {"type": type(value).__name__})
                print(f"[market-debug] event.{key} summary={_json(shape)}")
                if isinstance(value, dict):
                    for child_key, child in list(value.items())[:2]:
                        print(f"[market-debug] event.{key}[{_json(str(child_key))}] sample="
                              f"{_json(sanitize_identity(child))}")
                elif isinstance(value, list):
                    for index, child in enumerate(value[:2]):
                        print(f"[market-debug] event.{key}[{index}] sample="
                              f"{_json(sanitize_identity(child))}")
            self.snapshot_events.append({
                "eventID": _safe_scalar(event.get("eventID") or event.get("id")),
                "leagueID": _safe_scalar(event.get("leagueID")),
                "eventType": _safe_scalar(event.get("eventType") or event.get("type")),
                "status": sanitize_identity(event.get("status")),
                "identity": safe_identity,
                "odds": [],
            })

        odds = event.get("odds") or {}
        rows = odds.values() if isinstance(odds, dict) else odds if isinstance(odds, list) else []
        for odd in rows:
            if not isinstance(odd, dict):
                continue
            stat = str(odd.get("statID") or odd.get("marketName") or "").lower()
            is_player = bool(odd.get("playerID") or any(part in stat for part in _PLAYER_STAT_PARTS))
            if is_player and self.player_odds < MAX_PLAYER_ODDS:
                self.player_odds += 1
                print(f"[market-debug] player odd sample={_json(sanitize_odd(odd))}")
            elif not is_player and self.team_odds < MAX_TEAM_ODDS:
                self.team_odds += 1
                print(f"[market-debug] team odd sample={_json(sanitize_odd(odd))}")
            nested = odd.get("byBookmaker")
            if isinstance(nested, dict) and nested and self.book_rows < MAX_BOOK_ROWS:
                self.book_rows += 1
                book, row = next(((book, row) for book, row in nested.items()
                                  if isinstance(row, dict)), (None, {}))
                safe_row = {key: _safe_scalar(row.get(key)) for key in _BOOK_FIELDS if key in row}
                print(f"[market-debug] byBookmaker keys: {', '.join(list(nested)[:20])}")
                print(f"[market-debug] bookmaker row sample={_json({'bookmaker': book, **safe_row})}")
            if sample_event and len(self.snapshot_events[-1]["odds"]) < 10:
                self.snapshot_events[-1]["odds"].append(sanitize_odd(odd))

    def unrecognized_team(self, value) -> None:
        token = str(value or "").strip()
        if not self.enabled or not token or token in self.unknown_tokens or len(self.unknown_tokens) >= MAX_UNKNOWN_TOKENS:
            return
        self.unknown_tokens.add(token)
        print(f"[market-debug] unrecognized team token: {_safe_scalar(token)}")

    def team_resolution_failed(self, event: dict, details: dict) -> None:
        if not self.enabled or self.team_failures >= MAX_TEAM_FAILURES:
            return
        self.team_failures += 1
        print("[market-debug] event team resolution failed")
        print(f"[market-debug] eventID={_safe_scalar(event.get('eventID') or event.get('id'))}")
        print(f"[market-debug] team resolution={_json(sanitize_identity(details))}")
        print("[market-debug] provider request succeeded but team identity schema was not recognized")

    def write_snapshot(self) -> None:
        if not self.enabled or not self.snapshot_events:
            return
        try:
            payload = {"events": self.snapshot_events[:MAX_EVENTS]}
            encoded = _json(payload)
            if len(encoded) > 100_000:
                payload = {"events": [{**event, "identity": {"truncated": True},
                                        "odds": event["odds"][:5]}
                                       for event in self.snapshot_events[:MAX_EVENTS]]}
                encoded = _json(payload)
            self.snapshot_path.write_text(encoded)
            print(f"[market-debug] sanitized SportsGameOdds sample written to {self.snapshot_path}")
        except OSError as exc:
            print(f"[market-debug] sanitized sample write failed: {type(exc).__name__}")
