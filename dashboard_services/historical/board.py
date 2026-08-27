"""Compact board payload and deep-panel lookup (pure).

Phase 8 rides ``/api/league-players`` with a small ``historical`` dict per
skill player. The lazy deep panel reads named comps from JSON leaves. This
module does not scan parquet, fetch projections, or enter ranking / Pick Score.

Preseason matching fields for the *upcoming* season are derived from each
player's latest warehouse row (last observed season, not a fake calendar
join). Live board ADP / ``proj_ppg`` are caller-supplied. Live ``ppg``
(actuals) is never treated as a projection.

This module must stay dependency-free (no pandas, Flask, nfl_data_py, or I/O).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from dashboard_services.historical.comps import (
    extract_comp_query,
    lookup_board_probabilities,
)
from dashboard_services.historical.definitions import (
    SKILL_POSITIONS,
    display_percent,
    draft_capital_bucket,
    normalize_adp,
    _optional_float,
    _optional_int,
)
from dashboard_services.historical.signals import (
    compare_board_signals,
    lookup_market_probability,
    projected_ppg_of,
)

PRESEASON_FIELDS: tuple[str, ...] = (
    "position",
    "years_experience",
    "age",
    "draft_capital_bucket",
    "previous_season_finish",
    "previous_season_target_share",
    "previous_season_snap_pct",
    "previous_season_year",
)


def board_contract() -> dict:
    """JSON metadata for the cheat-sheet column and deep panel."""
    return {
        "descriptive_only": True,
        "not_in_ranking": True,
        "not_in_pick_score": True,
        "rides": "/api/league-players",
        "deep_panel": "/api/historical-player/<player_id>",
        "compact_fields": [
            "p_hit",
            "p_hit_pct",
            "mkt_p",
            "mkt_pct",
            "h_vs_m",
            "proj_rk",
            "adp_rk",
            "p_vs_m",
            "p_vs_h",
        ],
        "request_path": (
            "JSON preseason_profiles + live redraft ADP + resolver proj_ppg; "
            "no parquet scan, no new Postgres table"
        ),
        "adp_axis": "redraft_1qb",
        "sf_tep_historical": False,
    }


def build_preseason_profiles(
    rows: Sequence[Mapping[str, Any]],
    *,
    upcoming_season: Optional[int] = None,
) -> dict:
    """One preseason matching profile per player for the season after the warehouse.

    Last observed warehouse season supplies previous-season finish/usage.
    Years experience and Sept-1 age step forward by the calendar gap to
    ``upcoming_season`` (default: max warehouse season + 1). Missing dims stay
    omitted, never 0 / UDFA / last place.
    """
    latest: dict[str, dict] = {}
    max_season: Optional[int] = None
    for row in rows:
        season = _optional_int(row.get("season"))
        pid = str(row.get("sleeper_id") or "")
        pos = str(row.get("position") or "").upper()
        if season is None or not pid or pos not in SKILL_POSITIONS:
            continue
        max_season = season if max_season is None else max(max_season, season)
        prev = latest.get(pid)
        if prev is None or season > int(prev["season"]):
            latest[pid] = dict(row)

    if upcoming_season is None:
        upcoming_season = (max_season + 1) if max_season is not None else None

    profiles: dict[str, dict] = {}
    for pid, row in latest.items():
        last_season = _optional_int(row.get("season"))
        if last_season is None:
            continue
        gap = 1
        if upcoming_season is not None:
            gap = upcoming_season - last_season
            if gap < 0:
                gap = 0
        ye = _optional_int(row.get("years_experience"))
        draft_year = _optional_int(row.get("draft_year"))
        if draft_year is not None and upcoming_season is not None:
            new_ye = upcoming_season - draft_year
            if new_ye < 0:
                new_ye = None
        elif ye is not None:
            new_ye = ye + gap
        else:
            new_ye = None
        age = _optional_float(row.get("age"))
        new_age = round(age + gap, 1) if age is not None else None
        cap = row.get("draft_capital_bucket")
        if cap not in (None, ""):
            capital = cap
        else:
            capital = draft_capital_bucket(
                row.get("draft_round") or row.get("nfl_draft_round"),
                row.get("draft_pick") or row.get("nfl_draft_pick"),
                undrafted=bool(row.get("undrafted")),
            )
        rec = {
            "position": str(row.get("position") or "").upper(),
            "years_experience": new_ye,
            "age": new_age,
            "draft_capital_bucket": capital,
            "previous_season_finish": _optional_int(row.get("ppr_positional_finish")),
            "previous_season_target_share": _optional_float(row.get("target_share")),
            "previous_season_snap_pct": _optional_float(row.get("snap_pct")),
            "previous_season_year": last_season,
        }
        profiles[pid] = {k: v for k, v in rec.items() if v is not None}
    return {
        "upcoming_season": upcoming_season,
        "prior_season_floor": max_season,
        "n_players": len(profiles),
        "by_player": profiles,
    }


def live_redraft_adp(player: Mapping[str, Any]) -> Optional[float]:
    """Current redraft 1QB ADP. Dynasty / SF fields are not historical ADP."""
    candidates: list[Any] = [player.get("redraft_avg_pick"), player.get("adp_overall")]
    by = player.get("adp_by_source") or {}
    if isinstance(by, Mapping):
        for src in ("consensus", "sleeper", "mfl", "espn", "yahoo"):
            block = by.get(src)
            if isinstance(block, Mapping):
                candidates.append(block.get("redraft_avg_pick"))
            else:
                candidates.append(block)
    candidates.append(player.get("adp"))
    for raw in candidates:
        adp = normalize_adp(raw)
        if adp is not None:
            return adp
    return None


def query_for_board_player(
    player: Mapping[str, Any],
    profiles_by_player: Mapping[str, Mapping[str, Any]],
) -> dict:
    """Merge JSON preseason fields with live ADP / proj_ppg. No actuals."""
    pid = str(player.get("id") or player.get("sleeper_id") or player.get("player_id") or "")
    prior = dict(profiles_by_player.get(pid) or {})
    pos = str(player.get("position") or prior.get("position") or "").upper()
    query: dict[str, Any] = {"sleeper_id": pid}
    if pos in SKILL_POSITIONS:
        query["position"] = pos
    for key in PRESEASON_FIELDS:
        if key == "position":
            continue
        val = prior.get(key)
        if val is not None:
            query[key] = val
    if "years_experience" not in query:
        ye = _optional_int(player.get("years_exp") if player.get("years_exp") is not None else player.get("years_experience"))
        if ye is not None:
            query["years_experience"] = ye
    adp = live_redraft_adp(player)
    if adp is not None:
        query["adp_overall"] = adp
        query["adp"] = adp
    ppg = projected_ppg_of(player)
    if ppg is not None:
        query["projected_ppg"] = ppg
        query["proj_ppg"] = ppg
    return query


def compact_signal(full: Mapping[str, Any]) -> dict:
    """Board-sized slice. No named comps, no blended score."""
    history = full.get("history") if isinstance(full.get("history"), Mapping) else {}
    market = full.get("market") if isinstance(full.get("market"), Mapping) else {}
    projection = full.get("projection") if isinstance(full.get("projection"), Mapping) else {}
    comparison = full.get("comparison") if isinstance(full.get("comparison"), Mapping) else {}
    h_vs_m = comparison.get("history_vs_market") if isinstance(comparison.get("history_vs_market"), Mapping) else {}
    p_vs_m = comparison.get("projection_vs_market") if isinstance(comparison.get("projection_vs_market"), Mapping) else {}
    p_vs_h = comparison.get("projection_vs_history") if isinstance(comparison.get("projection_vs_history"), Mapping) else {}
    p_hit = history.get("p_top_12")
    mkt_p = market.get("p_top_12")
    return {
        "p_hit": p_hit,
        "p_hit_pct": display_percent(p_hit),
        "conf": history.get("confidence"),
        "n": history.get("sample_size"),
        "mkt_p": mkt_p,
        "mkt_pct": display_percent(mkt_p),
        "mkt_bucket": market.get("adp_bucket"),
        "h_vs_m": h_vs_m.get("label") or "unknown",
        "proj_rk": projection.get("implied_positional_rank"),
        "adp_rk": p_vs_m.get("adp_positional_rank"),
        "p_vs_m": p_vs_m.get("label") or "unknown",
        "p_vs_h": p_vs_h.get("label") or "unknown",
        "implies_top_12": projection.get("implies_top_12"),
    }


def attach_historical_signals(
    players: Sequence[Mapping[str, Any]],
    aggregates: Mapping[str, Any],
) -> list[dict]:
    """Stamp a compact ``historical`` dict on skill-position players in place.

    Non-skill rows are left untouched. Returns the compact list (same order).
    """
    pre = aggregates.get("preseason_profiles") or {}
    by_player = pre.get("by_player") if isinstance(pre, Mapping) else {}
    if not isinstance(by_player, Mapping):
        by_player = {}

    queries: list[dict] = []
    index_map: list[Optional[int]] = []
    for row in players:
        pos = str(row.get("position") or "").upper()
        if pos not in SKILL_POSITIONS:
            index_map.append(None)
            continue
        index_map.append(len(queries))
        queries.append(query_for_board_player(row, by_player))

    compared = compare_board_signals(queries, aggregates) if queries else []
    compact_out: list[dict] = []
    for i, row in enumerate(players):
        qi = index_map[i]
        if qi is None:
            compact_out.append({})
            continue
        compact = compact_signal(compared[qi])
        if isinstance(row, dict):
            row["historical"] = compact
        compact_out.append(compact)
    return compact_out


def build_deep_panel(
    player_id: str,
    aggregates: Mapping[str, Any],
    *,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Named comps + rates for the modal. JSON lookup only."""
    pid = str(player_id or "")
    pre = aggregates.get("preseason_profiles") or {}
    by_player = pre.get("by_player") if isinstance(pre, Mapping) else {}
    if not isinstance(by_player, Mapping):
        by_player = {}
    seed: dict[str, Any] = {"id": pid, "sleeper_id": pid}
    prior = by_player.get(pid) or {}
    if isinstance(prior, Mapping):
        seed.update(prior)
    if extra:
        seed.update(dict(extra))
        if extra.get("redraft_avg_pick") is not None or extra.get("adp") is not None:
            seed["adp_overall"] = live_redraft_adp(seed)
    query = query_for_board_player(seed, by_player)
    comps = aggregates.get("comps") if isinstance(aggregates.get("comps"), Mapping) else aggregates
    looked = lookup_board_probabilities(query, comps if isinstance(comps, Mapping) else {})
    market = lookup_market_probability(query, aggregates)
    return {
        "available": True,
        "player_id": pid,
        "descriptive_only": True,
        "no_blended_score": True,
        "not_in_ranking": True,
        "preseason": extract_comp_query(query),
        "history": {
            "n": looked.get("n"),
            "key_used": looked.get("key_used"),
            "dropped": looked.get("dropped"),
            "fallback": looked.get("fallback"),
            "rates": looked.get("rates"),
            "examples": looked.get("examples") or [],
            "kind": "conditional",
        },
        "market": market,
    }
