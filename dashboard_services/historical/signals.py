"""History vs Projection vs Market — three live signals, never blended (pure).

History is comps ``P(top-12)`` from precomputed JSON leaves. Market is
historical ``P(top-12 | current ADP overall bucket)``. Projection is the
caller's current Sleeper PPG plus an implied positional rank among the
**live projected field**. PPG is not converted into a probability.

Native units stay distinct:

* history vs market — probability
* projection vs market — positional rank
* projection vs history — qualitative only

Missing any input stays ``unknown`` / ``None``, never a fake 0% or last
place. This layer is descriptive and does not enter ranking or Pick Score.

Callers must resolve PPG via ``utils.projection_resolver`` (Sleeper is
authoritative). This module does not fetch, annualize, or backfill
projections, and it does not read warehouse ``ppr_ppg`` / board ``ppg`` as
a projection. Canonical warehouse rows have no ``projected_*`` columns.

Request path: pass the already-loaded ``historical_profile_aggregates.json``
plus the current board's live fields. No parquet scan, no new Postgres table.

This module must stay dependency-free (no pandas, Flask, nfl_data_py, or I/O).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from dashboard_services.historical.comps import lookup_board_probabilities
from dashboard_services.historical.career_path import apply_career_path_history
from dashboard_services.historical.definitions import (
    SIGNAL_BOARD_TIER,
    SIGNAL_HISTORY_BULLISH_P,
    SIGNAL_HISTORY_SKEPTICAL_P,
    SIGNAL_PROB_ALIGN_DELTA,
    SIGNAL_RANK_ALIGN_SPOTS,
    SKILL_POSITIONS,
    TIER_CUTOFFS,
    adp_overall_bucket,
    normalize_adp,
    _optional_float,
    _optional_int,
)
from dashboard_services.historical.finishes import competition_ranks

_ID_KEYS: tuple[str, ...] = ("sleeper_id", "player_id", "id")
_PROJ_PPG_KEYS: tuple[str, ...] = ("projected_ppg", "proj_ppg")
_ADP_KEYS: tuple[str, ...] = ("adp_overall", "adp", "overall_adp")
# Actuals / wrong units — never treat these as a current projection.
_NOT_PROJECTION_KEYS: tuple[str, ...] = (
    "ppg",
    "ppr_ppg",
    "half_ppr_ppg",
    "standard_ppg",
    "projected_points",
    "ppr_points",
)


def signal_contract() -> dict:
    """JSON metadata for the request-path comparison. No live numbers."""
    return {
        "descriptive_only": True,
        "no_blended_score": True,
        "no_historical_projection_backfill": True,
        "warehouse_has_projections": False,
        "request_path": (
            "compare_board_signals(board, historical_profile_aggregates.json); "
            "no parquet scan, no new Postgres table"
        ),
        "projection_resolver": "utils.projection_resolver (Sleeper-authoritative)",
        "native_units": {
            "history": "P(top-12) from comps leaves",
            "market": "P(top-12) from historical ADP overall bucket",
            "projection": (
                "current PPG + implied positional rank among the projected "
                "field; not a probability"
            ),
            "history_vs_market": "probability",
            "projection_vs_market": "positional_rank",
            "projection_vs_history": "qualitative_only",
        },
        "align": {
            "probability_delta": SIGNAL_PROB_ALIGN_DELTA,
            "rank_spots": SIGNAL_RANK_ALIGN_SPOTS,
            "history_skeptical_p": SIGNAL_HISTORY_SKEPTICAL_P,
            "history_bullish_p": SIGNAL_HISTORY_BULLISH_P,
            "tier": SIGNAL_BOARD_TIER,
        },
        "ignored_as_projection": list(_NOT_PROJECTION_KEYS),
    }


def normalize_projected_ppg(value: Any) -> Optional[float]:
    """Current PPG. Missing / non-positive → None, not 0."""
    ppg = _optional_float(value)
    if ppg is None or ppg <= 0:
        return None
    return ppg


def player_id_of(row: Mapping[str, Any], index: Optional[int] = None) -> str:
    for key in _ID_KEYS:
        val = row.get(key)
        if val is not None and str(val):
            return str(val)
    if index is not None:
        return f"row:{index}"
    return ""


def projected_ppg_of(row: Mapping[str, Any]) -> Optional[float]:
    """Live projection only. Warehouse actuals and season totals are ignored."""
    for key in _PROJ_PPG_KEYS:
        ppg = normalize_projected_ppg(row.get(key))
        if ppg is not None:
            return ppg
    return None


def overall_adp_of(row: Mapping[str, Any]) -> Optional[float]:
    for key in _ADP_KEYS:
        adp = normalize_adp(row.get(key))
        if adp is not None:
            return adp
    return None


def _position_of(row: Mapping[str, Any]) -> Optional[str]:
    pos = str(row.get("position") or row.get("pos") or "").upper()
    return pos if pos in SKILL_POSITIONS else None


def _round_p(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 6)


def probability_from_rate(rate: Any) -> Optional[float]:
    """Board-facing P from a rate record. Empty cohorts stay None, not 0%."""
    if not isinstance(rate, Mapping):
        return None
    if rate.get("raw_rate") is None:
        return None
    smoothed = rate.get("smoothed_rate")
    if smoothed is not None:
        return _round_p(smoothed)
    return _round_p(rate.get("raw_rate"))


def positional_ranks_for_board(
    players: Sequence[Mapping[str, Any]],
    *,
    values: Sequence[Optional[float]],
    higher_is_better: bool,
) -> list[Optional[int]]:
    """Competition rank (1, 2, 2, 4) within each position. Missing stays None."""
    if len(values) != len(players):
        raise ValueError("values must align with players")
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(players):
        pos = _position_of(row)
        if pos is None:
            continue
        groups.setdefault(pos, []).append(i)
    ranks: list[Optional[int]] = [None] * len(players)
    for idxs in groups.values():
        raw = [values[i] for i in idxs]
        keyed = [
            (v if higher_is_better else (-v if v is not None else None))
            for v in raw
        ]
        placed = competition_ranks(keyed)
        for i, rank in zip(idxs, placed):
            ranks[i] = rank
    return ranks


def implied_projection_ranks(players: Sequence[Mapping[str, Any]]) -> list[Optional[int]]:
    """Implied positional rank among players who have a current projected PPG."""
    return positional_ranks_for_board(
        players,
        values=[projected_ppg_of(row) for row in players],
        higher_is_better=True,
    )


def implied_adp_positional_ranks(players: Sequence[Mapping[str, Any]]) -> list[Optional[int]]:
    """Live positional ADP rank among the provided board (lower pick is better)."""
    return positional_ranks_for_board(
        players,
        values=[overall_adp_of(row) for row in players],
        higher_is_better=False,
    )


def lookup_history_probability(
    player: Mapping[str, Any],
    aggregates: Mapping[str, Any],
    *,
    tier: str = SIGNAL_BOARD_TIER,
) -> dict:
    """Comps P(hit) from JSON leaves. Does not scan parquet."""
    comps = aggregates.get("comps") if isinstance(aggregates.get("comps"), Mapping) else aggregates
    looked = lookup_board_probabilities(player, comps if isinstance(comps, Mapping) else {})
    looked = apply_career_path_history(player, looked, aggregates)
    rate = (looked.get("rates") or {}).get(tier) or {}
    p = probability_from_rate(rate)
    n = _optional_int(looked.get("n")) or 0
    unknown = "no_position" if not looked.get("position") else (
        "empty_cell" if p is None else None
    )
    return {
        "p_top_12": p,
        "raw_p_top_12": _round_p(rate.get("raw_rate")) if isinstance(rate, Mapping) else None,
        "sample_size": n,
        "confidence": rate.get("confidence") if isinstance(rate, Mapping) else None,
        "relaxed": bool(looked.get("fallback")),
        "dropped": list(looked.get("dropped") or []),
        "key_used": dict(looked.get("key_used") or {}),
        "source": looked.get("source") or "comps",
        "unknown_reason": unknown,
    }


def lookup_market_probability(
    player: Mapping[str, Any],
    aggregates: Mapping[str, Any],
    *,
    tier: str = SIGNAL_BOARD_TIER,
    overall_adp: Optional[float] = None,
) -> dict:
    """Historical P(hit | current overall ADP bucket). Missing ADP is unknown."""
    adp = normalize_adp(overall_adp) if overall_adp is not None else overall_adp_of(player)
    bucket = adp_overall_bucket(adp)
    pos = _position_of(player)
    adp_section = aggregates.get("adp") if isinstance(aggregates.get("adp"), Mapping) else {}
    by_pos = (adp_section.get("by_position") or {}) if isinstance(adp_section, Mapping) else {}
    node = (by_pos.get(pos) or {}) if pos else {}
    pair = (node.get("by_overall_bucket") or {}).get(bucket) if bucket else None
    conditional = (pair.get("conditional") or {}) if isinstance(pair, Mapping) else {}
    p = probability_from_rate(conditional)
    if adp is None or bucket is None:
        unknown = "missing_adp"
    elif not pos:
        unknown = "no_position"
    elif p is None:
        unknown = "empty_bucket"
    else:
        unknown = None
    return {
        "overall_adp": adp,
        "adp_bucket": bucket,
        "p_top_12": p,
        "raw_p_top_12": _round_p(conditional.get("raw_rate")) if conditional else None,
        "sample_size": _optional_int(conditional.get("sample_size")) or 0,
        "confidence": conditional.get("confidence") if conditional else None,
        "source": "adp_hit_rates",
        "unknown_reason": unknown,
    }


def _implies_top_n(rank: Optional[int], *, cutoff: int) -> Optional[bool]:
    if rank is None:
        return None
    return rank <= cutoff


def compare_history_vs_market(
    history_p: Optional[float],
    market_p: Optional[float],
    *,
    delta_threshold: float = SIGNAL_PROB_ALIGN_DELTA,
) -> dict:
    if history_p is None or market_p is None:
        return {
            "unit": "probability",
            "history_p": history_p,
            "market_p": market_p,
            "delta": None,
            "label": "unknown",
        }
    delta = _round_p(history_p - market_p)
    assert delta is not None
    if abs(delta) < delta_threshold:
        label = "aligned"
    elif delta >= delta_threshold:
        label = "history_higher"
    else:
        label = "market_higher"
    return {
        "unit": "probability",
        "history_p": history_p,
        "market_p": market_p,
        "delta": delta,
        "label": label,
    }


def compare_projection_vs_market(
    projected_rank: Optional[int],
    adp_positional_rank: Optional[int],
    *,
    rank_spots: int = SIGNAL_RANK_ALIGN_SPOTS,
) -> dict:
    if projected_rank is None or adp_positional_rank is None:
        return {
            "unit": "positional_rank",
            "projected_rank": projected_rank,
            "adp_positional_rank": adp_positional_rank,
            "delta": None,
            "label": "unknown",
        }
    # Positive delta = projection ranks the player better (lower number).
    delta = adp_positional_rank - projected_rank
    if abs(delta) <= rank_spots:
        label = "aligned"
    elif delta > rank_spots:
        label = "projection_higher"
    else:
        label = "market_higher"
    return {
        "unit": "positional_rank",
        "projected_rank": projected_rank,
        "adp_positional_rank": adp_positional_rank,
        "delta": delta,
        "label": label,
    }


def compare_projection_vs_history(
    implies_top_12: Optional[bool],
    history_p: Optional[float],
    *,
    skeptical_p: float = SIGNAL_HISTORY_SKEPTICAL_P,
    bullish_p: float = SIGNAL_HISTORY_BULLISH_P,
) -> dict:
    """Qualitative only — does not invent P(top-12) from PPG."""
    if implies_top_12 is None or history_p is None:
        label = "unknown"
    elif implies_top_12 and history_p < skeptical_p:
        label = "history_skeptical"
    elif (not implies_top_12) and history_p >= bullish_p:
        label = "history_bullish"
    elif implies_top_12:
        label = "agree_hit"
    else:
        label = "agree_miss"
    return {
        "unit": "qualitative",
        "implies_top_12": implies_top_12,
        "history_p": history_p,
        "label": label,
        "note": "PPG is not converted to P(top-12)",
    }


def compare_player_signals(
    player: Mapping[str, Any],
    aggregates: Mapping[str, Any],
    *,
    projected_positional_rank: Optional[int] = None,
    adp_positional_rank: Optional[int] = None,
    index: Optional[int] = None,
    tier: str = SIGNAL_BOARD_TIER,
) -> dict:
    """One player's three-signal comparison. ``blended_score`` is always None."""
    pid = player_id_of(player, index)
    pos = _position_of(player)
    cutoff = TIER_CUTOFFS.get(tier) or TIER_CUTOFFS[SIGNAL_BOARD_TIER]
    ppg = projected_ppg_of(player)
    if projected_positional_rank is None:
        projected_positional_rank = _optional_int(player.get("projected_positional_rank"))
    if adp_positional_rank is None:
        adp_positional_rank = _optional_int(
            player.get("adp_positional_rank") or player.get("adp_positional")
        )

    history = lookup_history_probability(player, aggregates, tier=tier)
    market = lookup_market_probability(player, aggregates, tier=tier)
    implies = _implies_top_n(projected_positional_rank, cutoff=cutoff)
    proj_unknown = None if ppg is not None else "missing_ppg"
    projection = {
        "ppg": ppg,
        "implied_positional_rank": projected_positional_rank,
        "implies_top_12": implies,
        "unit": "points_per_game",
        "source": "current_sleeper",
        "unknown_reason": proj_unknown,
    }
    return {
        "player_id": pid,
        "position": pos,
        "history": history,
        "market": market,
        "projection": projection,
        "comparison": {
            "history_vs_market": compare_history_vs_market(
                history.get("p_top_12"), market.get("p_top_12")
            ),
            "projection_vs_market": compare_projection_vs_market(
                projected_positional_rank, adp_positional_rank
            ),
            "projection_vs_history": compare_projection_vs_history(
                implies, history.get("p_top_12")
            ),
            "blended_score": None,
            "no_blended_score": True,
        },
    }


def compare_board_signals(
    players: Sequence[Mapping[str, Any]],
    aggregates: Mapping[str, Any],
    *,
    tier: str = SIGNAL_BOARD_TIER,
) -> list[dict]:
    """Batch comparison for a board. Implied ranks use the provided field.

    One call covers the whole board so positional rank is well-defined.
    Per-player REST is the deep panel (Phase 8), not this path.
    """
    proj_ranks = implied_projection_ranks(players)
    adp_ranks = implied_adp_positional_ranks(players)
    out: list[dict] = []
    for i, row in enumerate(players):
        out.append(
            compare_player_signals(
                row,
                aggregates,
                projected_positional_rank=proj_ranks[i],
                adp_positional_rank=adp_ranks[i],
                index=i,
                tier=tier,
            )
        )
    return out
