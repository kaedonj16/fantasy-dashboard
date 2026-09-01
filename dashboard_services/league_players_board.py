"""Compact /api/league-players payload for the Draft Cheat Sheet.

The full league-players blob is shared with Rankings and the Draft Room and
carries dynasty-size values, provenance, actuals, picks, and K/DEF. The cheat
sheet only ranks QB/RB/WR/TE by VOR and paints a handful of columns, so this
view drops everything else. Replacement level is computed in the browser from
every skill player with a value, so 0-value and non-skill rows are safe to omit.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional


SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}
VALUE_KEYS = ("redraft_value_1qb", "redraft_value_sf", "value", "sf_value")
ADP_AXES = ("avg_pick", "sf_avg_pick", "redraft_avg_pick", "sf_redraft_avg_pick")
PLAYER_KEYS = (
    "id",
    "name",
    "position",
    "age",
    "value",
    "sf_value",
    "redraft_value_1qb",
    "redraft_value_sf",
    "avg_pick",
    "sf_avg_pick",
    "redraft_avg_pick",
    "sf_redraft_avg_pick",
    "proj_ppg",
    "proj_ppg_by",
    "projection",
    "projected_offense_rank",
    "market_vs_adp",
    "market_expected_adp",
    "market_confidence",
    "market_confidence_label",
    "market_basis",
    "historical",
    "adp_by_source",
)


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _has_board_value(player: Mapping[str, Any]) -> bool:
    return any(_num(player.get(key)) > 0 for key in VALUE_KEYS)


def _slim_adp_by_source(by_source: Any) -> Optional[dict]:
    if not isinstance(by_source, Mapping):
        return None
    out = {}
    for source, block in by_source.items():
        if not isinstance(block, Mapping):
            continue
        slim_block = {
            axis: block[axis]
            for axis in ADP_AXES
            if block.get(axis) is not None
        }
        if slim_block:
            out[str(source)] = slim_block
    return out or None


def _market_vs_adp_of(player: Mapping[str, Any], *, is_superflex: bool) -> Any:
    if is_superflex:
        if player.get("sf_market_vs_adp") is not None:
            return player.get("sf_market_vs_adp")
    elif player.get("market_vs_adp_1qb") is not None:
        return player.get("market_vs_adp_1qb")
    return player.get("market_vs_adp")


def _slim_proj_ppg_by(by_variant: Any) -> Optional[dict]:
    """Keep scoring-variant PPG so the sheet can retarget PPR / TEP / 6-pt TD
    without waiting on a projection overlay rebuild."""
    if not isinstance(by_variant, Mapping):
        return None
    out = {}
    for key, value in by_variant.items():
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        if num:
            out[str(key)] = num
    return out or None


def _slim_projection(projection: Any) -> Optional[dict]:
    """Keep the canonical PPG stamp plus the variant it was scored for."""
    if not isinstance(projection, Mapping):
        return None
    compact = {
        key: projection[key]
        for key in ("ppg", "unit", "projection_type", "scoring_variant")
        if projection.get(key) is not None
    }
    if compact.get("ppg") is None:
        return None
    return compact


def slim_board_player(player: Mapping[str, Any], *, is_superflex: bool) -> Optional[dict]:
    """One cheat-sheet row, or None when the player cannot appear on the board."""
    pos = str(player.get("position") or "").upper()
    if pos not in SKILL_POSITIONS or not _has_board_value(player):
        return None
    row = {}
    for key in PLAYER_KEYS:
        if key == "adp_by_source":
            by_source = _slim_adp_by_source(player.get("adp_by_source"))
            if by_source:
                row[key] = by_source
            continue
        if key == "market_vs_adp":
            value = _market_vs_adp_of(player, is_superflex=is_superflex)
        elif key == "historical":
            value = player.get("historical")
            if not isinstance(value, Mapping) or not value:
                continue
            value = dict(value)
        elif key == "projection":
            value = _slim_projection(player.get("projection"))
        elif key == "proj_ppg_by":
            value = _slim_proj_ppg_by(player.get("proj_ppg_by"))
        else:
            value = player.get(key)
        if value is not None:
            row[key] = value
    if player.get("id") is not None:
        row["id"] = player.get("id")
    row["position"] = pos
    return row


def slim_board_payload(payload: Mapping[str, Any], *, is_superflex: bool) -> dict:
    """Return a cheat-sheet-sized copy. Does not mutate ``payload``."""
    players = []
    for player in payload.get("players") or []:
        if not isinstance(player, Mapping):
            continue
        row = slim_board_player(player, is_superflex=is_superflex)
        if row is not None:
            players.append(row)
    out = {"players": players}
    options = payload.get("adp_source_options")
    if options:
        out["adp_source_options"] = options
    out["market_vs_adp_available"] = any(
        player.get("market_vs_adp") is not None for player in players
    )
    out["historical_available"] = payload.get("historical_available") is True
    return out
