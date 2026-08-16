from __future__ import annotations

import re
import unicodedata


def normalize_player_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode().lower()
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", text)
    return re.sub(r"[^a-z0-9]", "", text)


def resolve_player(provider_id: str, name: str, position: str, team: str,
                   players: dict, persisted: dict[str, str] | None = None) -> tuple[str | None, float]:
    """Resolve once by constrained metadata. Ambiguous candidates fail closed."""
    if persisted and provider_id in persisted:
        return persisted[provider_id], 1.0
    wanted_name, wanted_pos = normalize_player_name(name), (position or "").upper()
    matches = []
    for player_id, meta in (players or {}).items():
        if normalize_player_name(meta.get("name") or meta.get("full_name") or "") != wanted_name:
            continue
        pos = str(meta.get("position") or meta.get("pos") or "").upper()
        if wanted_pos and pos != wanted_pos:
            continue
        matches.append((str(player_id), str(meta.get("team") or "").upper() == str(team or "").upper()))
    if len(matches) == 1:
        return matches[0][0], 0.95 if matches[0][1] else 0.88
    team_matches = [pid for pid, same_team in matches if same_team]
    return (team_matches[0], 0.9) if len(team_matches) == 1 else (None, 0.0)
