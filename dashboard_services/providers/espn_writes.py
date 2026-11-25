"""ESPN write client for Autopilot: lineup transactions only.

Productionizes the Phase 0 probe (~/workspace/autopilot-probe/espn_probe.py,
proven live 2026-11-24 on league 887776065): POST a ROSTER transaction with
LINEUP items to lm-api-writes, success = HTTP 2xx + body status "EXECUTED",
verify by re-reading the roster from lm-api-reads.

Hard rules baked into this module:

- Exactly two operations exist: ``submit_lineup`` and ``read_lineup_state``
  (plus the payload builder and verify helper they use). There is no
  waiver, add/drop, or trade function here, and there never will be
  (guardrail 1; waivers are a separate Phase 1.5 probe).
- Credentials resolve ONLY for the acting account: the caller passes an
  explicit ``account_id`` and resolution goes through
  ``accounts.get_espn_league_credentials(account_id, ...)`` with the
  same-account fallback ``accounts.get_any_espn_account_credentials``,
  mirroring ``espn_api._resolve_espn_request_creds``. There is no env
  fallback and no cross-account scan: writing with another league member's
  stored cookies is a hard failure, never a fallback.
- A rejected write raises :class:`EspnWriteRejected` with ESPN's status and
  messages verbatim. Nothing in this module retries a write. (The probe's
  revert-retry existed to undo a test swap; Phase 1 never writes
  speculatively, so no revert concept exists here.)
- Reads are direct lm-api-reads calls with the acting account's cookies:
  no espn_api League-object caching sits between a write and its verify
  re-read, so verification always sees post-write reality.

Transport is injectable (``transport=``) so tests run against fakes shaped
from the captured real traffic; the default transport uses ``requests``
exactly like the probe.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

READ_URL = (
    "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/"
    "seasons/{season}/segments/0/leagues/{league_id}"
)
WRITE_URL = (
    "https://lm-api-writes.fantasy.espn.com/apis/v3/games/ffl/"
    "seasons/{season}/segments/0/leagues/{league_id}/transactions/"
)

BENCH_SLOT = 20
IR_SLOT = 21
FLEX_SLOT = 23

# Same browser fingerprint espn_api.py sends, plus the x-fantasy-* headers
# the real ESPN web client sends (probe-proven set).
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://fantasy.espn.com/",
}
WRITE_HEADERS = {
    "Content-Type": "application/json",
    "x-fantasy-source": "kona",
    "x-fantasy-platform": "kona-PROD",
}

REQUEST_TIMEOUT = 15

#: ESPN defaultPositionId -> site position name.
ESPN_POSITION_NAMES = {
    1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DEF",
    23: "QB",  # TQB leagues roster a team QB; treat as QB for planning
}


class AutopilotCredentialError(Exception):
    """The acting account has no usable ESPN credentials for this league."""


class EspnWriteRejected(Exception):
    """ESPN refused a lineup write. Carries the verbatim status/messages."""

    def __init__(self, http_status: int, body: Dict[str, Any]):
        self.http_status = http_status
        self.body = body or {}
        self.espn_status = self.body.get("status")
        super().__init__(describe_rejection(http_status, self.body))


class EspnReadError(Exception):
    """A roster read failed or the acting account's team was unresolvable."""


def describe_rejection(http_status: int, body: Dict[str, Any]) -> str:
    parts = [f"HTTP {http_status}", f"status={body.get('status')!r}"]
    for msg in body.get("messages") or []:
        parts.append(f"message: {msg}")
    for det in body.get("details") or []:
        if isinstance(det, dict):
            parts.append(
                f"detail[{det.get('type')}]: {det.get('message') or det.get('shortMessage')}"
            )
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Credentials: acting account only, ever.
# ---------------------------------------------------------------------------

def resolve_account_credentials(account_id, league_id, season) -> Dict[str, str]:
    """Resolve espn_s2 + SWID for THIS account and league, or raise.

    Mirrors espn_api._resolve_espn_request_creds minus the session and env
    paths: league-attached credentials first, then any ESPN login on the
    same account. Never another account's credentials.
    """
    if not account_id:
        raise AutopilotCredentialError("No acting account given for ESPN write.")
    from dashboard_services.accounts import (
        get_any_espn_account_credentials,
        get_espn_league_credentials,
    )

    creds = get_espn_league_credentials(int(account_id), str(league_id), int(season)) or {}
    if not (creds.get("espn_s2") and creds.get("swid")):
        creds = get_any_espn_account_credentials(int(account_id)) or {}
    espn_s2 = str(creds.get("espn_s2") or "").strip()
    swid = str(creds.get("swid") or "").strip()
    if not (espn_s2 and swid):
        raise AutopilotCredentialError(
            f"No ESPN credentials stored for account {account_id} "
            f"(league {league_id}). Autopilot cannot write for this league."
        )
    return {"espn_s2": espn_s2, "swid": swid}


def _normalize_swid(swid: str) -> str:
    """Match accounts.normalize_espn_swid: strip braces, uppercase."""
    s = str(swid or "").strip()
    if len(s) >= 2 and s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    return s.strip("{}").upper()


def _braced_swid(swid: str) -> str:
    s = str(swid or "").strip()
    if s and not (s.startswith("{") and s.endswith("}")):
        s = "{" + s.strip("{}") + "}"
    return s


# ---------------------------------------------------------------------------
# Transport (default: requests, like the probe; tests inject a fake)
# ---------------------------------------------------------------------------

class EspnTransport:
    """Direct lm-api-reads / lm-api-writes transport for one credential set."""

    def __init__(self, espn_s2: str, swid: str):
        import requests

        self._requests = requests
        self._cookies = {"SWID": _braced_swid(swid), "espn_s2": espn_s2}

    def get_league(self, season: int, league_id: str, params) -> dict:
        url = READ_URL.format(season=int(season), league_id=int(league_id))
        resp = self._requests.get(
            url,
            params=params,
            cookies=self._cookies,
            headers=BROWSER_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code in (401, 403):
            raise EspnReadError(
                f"ESPN denied the roster read (HTTP {resp.status_code}). "
                "The stored session is likely expired; reconnect the league."
            )
        if not resp.ok:
            raise EspnReadError(f"ESPN roster read failed (HTTP {resp.status_code}).")
        data = resp.json()
        if not isinstance(data, dict):
            raise EspnReadError("ESPN roster read returned a non-object payload.")
        return data

    def post_transaction(self, season: int, league_id: str, payload: dict):
        """Return (http_status, body_dict). Never raises for HTTP errors:
        the caller must see ESPN's rejection verbatim."""
        url = WRITE_URL.format(season=int(season), league_id=int(league_id))
        headers = dict(BROWSER_HEADERS)
        headers.update(WRITE_HEADERS)
        resp = self._requests.post(
            url,
            data=json.dumps(payload),
            cookies=self._cookies,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        try:
            body = resp.json()
        except ValueError:
            body = {"messages": [f"Non-JSON response body (HTTP {resp.status_code})"]}
        if not isinstance(body, dict):
            body = {"messages": [str(body)]}
        return resp.status_code, body


def _transport_for(account_id, league_id, season, transport=None):
    """Return (transport, creds); resolves creds for the acting account."""
    creds = resolve_account_credentials(account_id, league_id, season)
    if transport is None:
        transport = EspnTransport(creds["espn_s2"], creds["swid"])
    return transport, creds


# ---------------------------------------------------------------------------
# Roster parsing: shapes follow espn_api.py's own parsing of mRoster/mTeam
# (and the probe): teams[].roster.entries[] carry playerId + lineupSlotId,
# playerPoolEntry.player{id,fullName,eligibleSlots,defaultPositionId,
# injuryStatus}; ownership is teams[].owners (SWID strings); the current
# period is the payload's top-level scoringPeriodId.
# ---------------------------------------------------------------------------

def current_scoring_period(payload: dict) -> int:
    period = payload.get("scoringPeriodId")
    try:
        period = int(period)
    except (TypeError, ValueError):
        period = 0
    if period <= 0:
        status = payload.get("status") or {}
        cmp_ = status.get("currentMatchupPeriod") or {}
        try:
            period = int(cmp_.get("id") or 0)
        except (TypeError, ValueError):
            period = 0
    return period


def find_team(payload: dict, swid: str) -> dict:
    """Resolve the acting account's team by SWID ownership."""
    mine = _normalize_swid(swid)
    for team in payload.get("teams") or []:
        if not isinstance(team, dict):
            continue
        owners = [str(o) for o in (team.get("owners") or []) if o]
        if any(_normalize_swid(o) == mine for o in owners):
            return team
    raise EspnReadError(
        "Could not resolve the acting account's team: no team's owners list "
        "contains the stored SWID."
    )


def team_name(team: dict) -> str:
    name = str(team.get("name") or "").strip()
    if name:
        return name
    loc = str(team.get("location") or "").strip()
    nick = str(team.get("nickname") or "").strip()
    return f"{loc} {nick}".strip()


def parse_entries(team: dict) -> List[Dict[str, Any]]:
    entries = []
    for entry in ((team.get("roster") or {}).get("entries") or []):
        if not isinstance(entry, dict):
            continue
        ppe = entry.get("playerPoolEntry") or {}
        player = ppe.get("player") or {}
        pid = entry.get("playerId")
        if pid is None:
            pid = ppe.get("id") or player.get("id")
        slot = entry.get("lineupSlotId")
        if slot is None:
            slot = ppe.get("lineupSlotId")
        try:
            pid_i, slot_i = int(pid), int(slot)
        except (TypeError, ValueError):
            continue
        locked = entry.get("lineupLocked")
        if locked is None:
            locked = ppe.get("lineupLocked")
        eligible = []
        for s in player.get("eligibleSlots") or []:
            try:
                eligible.append(int(s))
            except (TypeError, ValueError):
                continue
        pos_id = player.get("defaultPositionId")
        try:
            pos_id = int(pos_id)
        except (TypeError, ValueError):
            pos_id = None
        entries.append({
            "player_id": pid_i,
            "name": str(player.get("fullName") or f"player {pid_i}"),
            "slot": slot_i,
            "locked": bool(locked),
            "eligible_slots": eligible,
            "position_id": pos_id,
            "position": ESPN_POSITION_NAMES.get(pos_id) if pos_id is not None else None,
            "injury_status": str(player.get("injuryStatus") or ""),
            "team": str(player.get("proTeamId") or ""),
        })
    return entries


def snapshot_of(entries: Sequence[Dict[str, Any]]) -> Dict[int, int]:
    return {e["player_id"]: e["slot"] for e in entries}


# ---------------------------------------------------------------------------
# The two operations
# ---------------------------------------------------------------------------

def read_league_payload(
    account_id, league_id, season, *, views=("mTeam", "mSettings", "mRoster"),
    scoring_period_id: Optional[int] = None, transport=None,
) -> Tuple[dict, Dict[str, str]]:
    """Raw league payload read with the acting account's credentials."""
    transport, creds = _transport_for(account_id, league_id, season, transport)
    params = [("view", v) for v in views]
    if scoring_period_id:
        params.append(("scoringPeriodId", int(scoring_period_id)))
    payload = transport.get_league(int(season), str(league_id), params)
    if str(payload.get("id") or "") not in ("", str(league_id)):
        raise EspnReadError(
            f"League payload id {payload.get('id')!r} != requested {league_id}."
        )
    return payload, creds


def read_lineup_state(
    account_id, league_id, season, *, scoring_period_id: Optional[int] = None,
    transport=None,
) -> Dict[str, Any]:
    """The acting account's current lineup state for one scoring period.

    Returns {team_id, team_name, scoring_period_id, entries, snapshot}.
    The read is week-scoped (an unscoped bulk mRoster read can zero
    lineupSlotId values; espn_api documents the same hazard).
    """
    payload, creds = read_league_payload(
        account_id, league_id, season, transport=transport,
    )
    period = int(scoring_period_id or 0) or current_scoring_period(payload)
    if period <= 0:
        raise EspnReadError("Could not determine the current scoring period.")
    scoped, _ = read_league_payload(
        account_id, league_id, season, scoring_period_id=period,
        transport=transport,
    )
    team = find_team(scoped, creds["swid"])
    entries = parse_entries(team)
    if not entries:
        team = find_team(payload, creds["swid"])
        entries = parse_entries(team)
    if not entries:
        raise EspnReadError("Roster read returned no entries for the acting account's team.")
    return {
        "team_id": int(team.get("id")),
        "team_name": team_name(team),
        "scoring_period_id": period,
        "entries": entries,
        "snapshot": snapshot_of(entries),
        "payload": scoped,
    }


def build_lineup_payload(team_id: int, swid: str, scoring_period_id: int,
                         moves: Sequence[Dict[str, Any]]) -> dict:
    """The ONLY transaction this module builds: ROSTER / LINEUP moves.

    moves: [{player_id, from_slot, to_slot}, ...]. Shape is the captured
    real traffic the probe proved (envelope type ROSTER, items type LINEUP).
    """
    return {
        "isLeagueManager": False,
        "teamId": int(team_id),
        "type": "ROSTER",
        "memberId": _braced_swid(swid),
        "scoringPeriodId": int(scoring_period_id),
        "executionType": "EXECUTE",
        "items": [
            {
                "playerId": int(m["player_id"]),
                "type": "LINEUP",
                "fromLineupSlotId": int(m["from_slot"]),
                "toLineupSlotId": int(m["to_slot"]),
            }
            for m in moves
        ],
    }


def submit_lineup(
    account_id, league_id, season, team_id: int, scoring_period_id: int,
    moves: Sequence[Dict[str, Any]], *, transport=None,
) -> Dict[str, Any]:
    """POST one ROSTER/LINEUP transaction. Raises EspnWriteRejected on any
    refusal (non-2xx or status != EXECUTED); never retries."""
    if not moves:
        raise ValueError("submit_lineup requires at least one move")
    transport, creds = _transport_for(account_id, league_id, season, transport)
    payload = build_lineup_payload(team_id, creds["swid"], scoring_period_id, moves)
    http_status, body = transport.post_transaction(int(season), str(league_id), payload)
    if not (200 <= http_status < 300 and body.get("status") == "EXECUTED"):
        raise EspnWriteRejected(http_status, body)
    return {
        "http_status": http_status,
        "status": body.get("status"),
        "transaction_id": body.get("id"),
        "payload": payload,
    }


def verify_lineup(
    account_id, league_id, season, expected_snapshot: Dict[int, int], *,
    scoring_period_id: Optional[int] = None, transport=None,
) -> Tuple[bool, Dict[int, int]]:
    """Re-read the roster and compare against the intended snapshot.

    Returns (matches, actual_snapshot). Only the players named in
    ``expected_snapshot`` are compared when it is a partial expectation;
    callers pass the full intended snapshot for an exact check.
    """
    state = read_lineup_state(
        account_id, league_id, season,
        scoring_period_id=scoring_period_id, transport=transport,
    )
    actual = state["snapshot"]
    expected = {int(k): int(v) for k, v in (expected_snapshot or {}).items()}
    if set(expected) == set(actual):
        return actual == expected, actual
    subset_ok = all(actual.get(pid) == slot for pid, slot in expected.items())
    return subset_ok, actual
