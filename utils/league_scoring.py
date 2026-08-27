"""Provider-agnostic normalized league-scoring contract."""
from __future__ import annotations

import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)

_ALIASES = {
    "rec": ("rec", "pointsPerReception"),
    "bonus_rec_te": ("bonus_rec_te",),
    "pass_yd": ("pass_yd", "passYards"),
    "pass_td": ("pass_td", "passTD"),
    "pass_int": ("pass_int", "passInterceptions"),
    "rush_yd": ("rush_yd", "rushYards"),
    "rush_td": ("rush_td", "rushTD"),
    "rec_yd": ("rec_yd", "receivingYards"),
    "rec_td": ("rec_td", "receivingTD"),
    "fum_lost": ("fum_lost", "fumbles"),
}
_DEFAULTS = {
    "bonus_rec_te": 0.0, "pass_yd": 0.04, "pass_td": 4.0,
    "pass_int": -2.0, "rush_yd": 0.1, "rush_td": 6.0,
    "rec_yd": 0.1, "rec_td": 6.0, "fum_lost": -2.0,
}


def normalize_league_scoring(platform: str, raw_provider_settings: Mapping[str, Any] | None,
                             *, league_id=None, season=None) -> dict[str, Any]:
    """Return one canonical scoring shape while preserving explicit zeroes.

    Unknown provider-specific fields are retained so already-supported custom
    categories continue to flow into ``score_stats`` by exact key.
    """
    raw = dict(raw_provider_settings or {})
    out = dict(raw)
    for canonical, aliases in _ALIASES.items():
        value = next((raw[key] for key in aliases if key in raw and raw[key] is not None), None)
        if value is None:
            if canonical == "rec":
                # Documented conservative provider fallback, with visibility;
                # this is not confused with an explicitly configured zero.
                logger.warning("[league-scoring] missing reception scoring platform=%s "
                               "league_id=%s season=%s; conservative rec=0 fallback",
                               platform, league_id, season)
                value = 0.0
            else:
                value = _DEFAULTS.get(canonical)
        if value is not None:
            try:
                out[canonical] = float(value)
            except (TypeError, ValueError):
                logger.warning("[league-scoring] invalid %s platform=%s league_id=%s",
                               canonical, platform, league_id)
                if canonical in _DEFAULTS:
                    out[canonical] = _DEFAULTS[canonical]
    return out
