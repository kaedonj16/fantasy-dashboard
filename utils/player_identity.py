"""Conservative cross-provider NFL player identity resolution.

The site canonical player key is the key in ``players_index`` (normally a
Sleeper id).  This resolver is deliberately data-injected: provider adapters
remain authoritative for their own IDs and RedZone supplies the already-loaded
player index.  Missing or conflicting evidence returns unresolved/ambiguous;
it never guesses a fantasy actor.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Any, Mapping

from utils.nfl_stadiums import normalize_nfl_team

_ID_FIELDS = (
    "player_id", "sleeper_id", "tank01_id", "espn_id", "yahoo_id",
    "mfl_id", "fleaflicker_id", "gsis_id", "sportradar_id",
)
_ROLE_POSITIONS = {
    "passer": {"QB"}, "sack_victim": {"QB"}, "receiver": {"WR", "TE", "RB"},
    "target": {"WR", "TE", "RB"}, "rusher": {"QB", "RB", "WR"},
    "fumbler": {"QB", "RB", "WR", "TE"}, "kicker": {"K"},
}
# Deliberately tiny, verified football-name crosswalk.  These are not fuzzy
# matches: aliases are only considered with the same normalized surname, team,
# and role/position evidence, and ambiguity still fails closed.
_VERIFIED_GIVEN_ALIASES = {"ken": "kenneth", "kenneth": "ken"}


def normalize_player_name(value: Any) -> str:
    value = unicodedata.normalize("NFKD", str(value or ""))
    value = "".join(c for c in value if not unicodedata.combining(c)).lower()
    value = re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


class PlayerIdentityResolver:
    """Immutable indexes over one canonical player map."""

    def __init__(self, players: Mapping[str, Mapping[str, Any]] | None):
        self.players = {str(k): dict(v) for k, v in (players or {}).items() if isinstance(v, Mapping)}
        self.by_id: dict[str, set[str]] = defaultdict(set)
        self.by_name: dict[str, set[str]] = defaultdict(set)
        for canonical_id, meta in self.players.items():
            self.by_id[canonical_id].add(canonical_id)
            for field in _ID_FIELDS:
                raw = meta.get(field)
                if raw not in (None, ""):
                    self.by_id[str(raw)].add(canonical_id)
            name = normalize_player_name(meta.get("name") or meta.get("full_name"))
            if name:
                self.by_name[name].add(canonical_id)

    def _result(self, pid: str = "", *, confidence: str, method: str) -> dict:
        meta = self.players.get(pid) or {}
        return {
            "canonical_player_id": pid,
            "name": meta.get("name") or meta.get("full_name") or "",
            "team": normalize_nfl_team(meta.get("team")),
            "position": str(meta.get("pos") or meta.get("position") or "").upper(),
            "confidence": confidence,
            "resolution_method": method,
        }

    def resolve(self, *, provider: str = "", provider_player_id: Any = None,
                tank01_id: Any = None, name: Any = None, team: Any = None,
                position: Any = None, role: str = "") -> dict:
        for method, raw in (("tank01_id", tank01_id), ("provider_id", provider_player_id)):
            if raw not in (None, ""):
                matches = self.by_id.get(str(raw), set())
                if len(matches) == 1:
                    return self._result(next(iter(matches)), confidence="exact", method=method)
                if len(matches) > 1:
                    return self._result(confidence="ambiguous", method=method)

        wanted_name = normalize_player_name(name)
        candidates = set(self.by_name.get(wanted_name, set()))
        wanted_team = normalize_nfl_team(team)
        wanted_pos = str(position or "").upper()
        role_positions = _ROLE_POSITIONS.get(str(role or "").lower())
        if wanted_team:
            candidates = {p for p in candidates if normalize_nfl_team(self.players[p].get("team")) == wanted_team}
        if wanted_pos:
            candidates = {p for p in candidates if str(self.players[p].get("pos") or self.players[p].get("position") or "").upper() == wanted_pos}
        elif role_positions:
            candidates = {p for p in candidates if str(self.players[p].get("pos") or self.players[p].get("position") or "").upper() in role_positions}
        if len(candidates) == 1:
            method = "team_name_position" if (wanted_pos or role_positions) else "team_name"
            return self._result(next(iter(candidates)), confidence="strong", method=method)
        if len(candidates) > 1:
            return self._result(confidence="ambiguous", method="name_collision")

        parts = wanted_name.split()
        if len(parts) >= 2 and parts[0] in _VERIFIED_GIVEN_ALIASES and wanted_team:
            alias_name = " ".join([_VERIFIED_GIVEN_ALIASES[parts[0]], *parts[1:]])
            alias_candidates = set(self.by_name.get(alias_name, set()))
            alias_candidates = {
                p for p in alias_candidates
                if normalize_nfl_team(self.players[p].get("team")) == wanted_team
                and (not role_positions or str(self.players[p].get("pos") or self.players[p].get("position") or "").upper() in role_positions)
            }
            if len(alias_candidates) == 1:
                return self._result(next(iter(alias_candidates)), confidence="strong", method="verified_given_alias")
            if len(alias_candidates) > 1:
                return self._result(confidence="ambiguous", method="verified_alias_collision")

        # Initial+surnames are accepted only inside a known team and role/position.
        if wanted_team and len(parts) >= 2 and len(parts[0]) == 1:
            suffix = parts[-1]
            matches = []
            for pid, meta in self.players.items():
                pname = normalize_player_name(meta.get("name") or meta.get("full_name")).split()
                ppos = str(meta.get("pos") or meta.get("position") or "").upper()
                if (len(pname) >= 2 and pname[0].startswith(parts[0]) and pname[-1] == suffix
                        and normalize_nfl_team(meta.get("team")) == wanted_team
                        and (not role_positions or ppos in role_positions)):
                    matches.append(pid)
            if len(matches) == 1:
                return self._result(matches[0], confidence="fallback", method="team_initial_surname_role")
            if len(matches) > 1:
                return self._result(confidence="ambiguous", method="initial_surname_collision")
        return self._result(confidence="unresolved", method="no_match")


def resolve_player_identity(players, **evidence) -> dict:
    return PlayerIdentityResolver(players).resolve(**evidence)
