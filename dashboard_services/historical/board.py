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
    COMP_BOARD_TIERS,
    COMP_DIMENSION_ORDER,
    SKILL_POSITIONS,
    display_percent,
    draft_capital_bucket,
    integer_age,
    normalize_adp,
    _optional_float,
    _optional_int,
)
from dashboard_services.historical.signals import (
    compare_board_signals,
    compare_projection_vs_history,
    compare_projection_vs_market,
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

# Modal copy only. Matching still uses the snake_case keys in comps.
COMP_FEATURE_LABELS: dict[str, str] = {
    "position": "Position",
    "career_stage": "Career stage",
    "draft_capital": "Draft capital",
    "prior_finish": "Last year finish",
    "age_bucket": "Age",
    "target_share": "Last year target share",
    "snap_pct": "Last year snaps",
}
CAREER_STAGE_DISPLAY: dict[str, str] = {
    "rookie": "Rookie",
    "year_2": "Year 2",
    "year_3": "Year 3",
    "year_4": "Year 4",
    "year_5": "Year 5",
    "year_6_plus": "Year 6+",
}
DRAFT_CAPITAL_DISPLAY: dict[str, str] = {
    "round_1": "Round 1",
    "day_2": "Day 2 (rounds 2–3)",
    "day_3": "Day 3 (rounds 4–7)",
    "undrafted": "Undrafted",
}
PRIOR_FINISH_DISPLAY: dict[str, str] = {
    "none": "No prior season",
    "top_5": "Top 5",
    "top_12": "Top 12",
    "top_24": "Top 24",
    "top_36": "Top 36",
    "outside_36": "Outside top 36",
}
PRIOR_FINISH_TREND: dict[str, str] = {
    "none": "no prior season",
    "top_5": "top-5",
    "top_12": "top-12",
    "top_24": "top-24",
    "top_36": "top-36",
    "outside_36": "outside the top 36",
}
ADP_BUCKET_DISPLAY: dict[str, str] = {
    "round_1": "Round 1",
    "round_2": "Round 2",
    "round_3": "Round 3",
    "round_4": "Round 4",
    "round_5": "Round 5",
    "rounds_6_7": "Rounds 6–7",
    "rounds_8_10": "Rounds 8–10",
    "rounds_11_plus": "Rounds 11+",
}
TIER_FINISH_DISPLAY: dict[str, str] = {
    "top_5": "top-5",
    "top_12": "top-12",
    "top_24": "top-24",
}
CONFIDENCE_DISPLAY: dict[str, str] = {
    "low": "small sample",
    "moderate": "moderate sample",
    "good": "solid sample",
    "strong": "large sample",
}


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
    proj_rk = _optional_int(
        player.get("projected_positional_rank") or player.get("proj_rk")
    )
    if proj_rk is not None:
        query["projected_positional_rank"] = proj_rk
        query["proj_rk"] = proj_rk
    adp_rk = _optional_int(
        player.get("adp_positional_rank") or player.get("adp_rk")
    )
    if adp_rk is not None:
        query["adp_positional_rank"] = adp_rk
        query["adp_rk"] = adp_rk
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
        "mkt_sentence": format_market_sentence(market),
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


def _title_from_key(key: str) -> str:
    return str(key or "").replace("_", " ").strip().title()


def format_age_bucket_label(value: Any) -> str:
    """Turn warehouse age bins into readable age copy."""
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("<="):
        years = text[2:].strip()
        return f"{years} or younger" if years else text
    if text.endswith("+") and text[:-1].replace(".", "", 1).isdigit():
        return f"{text[:-1]} or older"
    return text.replace("-", "–")


def format_comp_bucket_value(dim: str, value: Any) -> str:
    """Human value for one matching dimension. Missing stays empty."""
    if value is None or value == "":
        return ""
    text = str(value)
    if dim == "position":
        return text.upper()
    if dim == "career_stage":
        return CAREER_STAGE_DISPLAY.get(text, _title_from_key(text))
    if dim == "draft_capital":
        return DRAFT_CAPITAL_DISPLAY.get(text, _title_from_key(text))
    if dim == "prior_finish":
        return PRIOR_FINISH_DISPLAY.get(text, _title_from_key(text))
    if dim == "age_bucket":
        return format_age_bucket_label(text)
    return text


def format_adp_bucket_label(bucket: Any) -> str:
    if bucket in (None, ""):
        return ""
    text = str(bucket)
    return ADP_BUCKET_DISPLAY.get(text, _title_from_key(text))


def _confidence_label(confidence: Any) -> Optional[str]:
    if confidence in (None, ""):
        return None
    text = str(confidence)
    return CONFIDENCE_DISPLAY.get(text, text.replace("_", " "))


def _sample_clause(n: Any, confidence: Any) -> str:
    bits: list[str] = []
    sample = _optional_int(n)
    if sample is not None:
        bits.append(f"n={sample}")
    conf = _confidence_label(confidence)
    if conf:
        bits.append(conf)
    if not bits:
        return ""
    return ", ".join(bits)


def format_market_sentence(
    market: Optional[Mapping[str, Any]],
    *,
    missing: str = "none",
) -> Optional[str]:
    """ADP-bucket hit-rate sentence. None when there is no rate to show.

    ``missing='no_adp'`` is for the lazy modal, which should explain a blank
    instead of looking empty. Compact board rows omit that copy.
    """
    mkt = market if isinstance(market, Mapping) else {}
    bucket_label = format_adp_bucket_label(mkt.get("adp_bucket"))
    mkt_pct = display_percent(mkt.get("p_top_12"))
    sample_bit = _sample_clause(mkt.get("sample_size"), mkt.get("confidence"))
    if mkt_pct is not None and bucket_label:
        sentence = (
            f"Players drafted in {bucket_label} historically finished top-12 "
            f"{mkt_pct}% of the time"
        )
        if sample_bit:
            sentence += f" ({sample_bit})"
        return sentence + "."
    if bucket_label:
        return (
            f"ADP is in {bucket_label}, but that historical bucket has no "
            "top-12 hit rate yet."
        )
    if missing == "no_adp":
        return (
            "This sheet has no live ADP for this player, so there is no "
            "ADP-bucket hit rate."
        )
    return None


def _as_rate(rec: Any) -> dict:
    if not isinstance(rec, Mapping):
        return {}
    if rec.get("display_pct") is not None or rec.get("sample_size") is not None:
        return dict(rec)
    cond = rec.get("conditional")
    if isinstance(cond, Mapping):
        return dict(cond)
    return {}


def _trend_row(
    *,
    kind: str,
    label: str,
    bucket: str,
    sentence: str,
    rate: Any,
) -> Optional[dict]:
    rec = _as_rate(rate)
    pct = rec.get("display_pct")
    if pct is None:
        return None
    return {
        "kind": kind,
        "label": label,
        "bucket": bucket,
        "sentence": sentence,
        "pct": pct,
        "n": rec.get("sample_size"),
        "confidence": rec.get("confidence"),
        "confidence_label": _confidence_label(rec.get("confidence")),
    }


def cohort_sentence(key_used: Optional[Mapping[str, Any]]) -> str:
    """Human description of the similar-profile group that produced Hist."""
    key = key_used if isinstance(key_used, Mapping) else {}
    pos = str(key.get("position") or "").upper() or "skill players"
    extras: list[str] = []
    stage = format_comp_bucket_value("career_stage", key.get("career_stage")) if key.get("career_stage") else ""
    cap = format_comp_bucket_value("draft_capital", key.get("draft_capital")) if key.get("draft_capital") else ""
    if stage and cap:
        extras.append(f"{stage.lower()} {cap.lower()} capital")
    elif stage:
        extras.append(stage.lower())
    elif cap:
        extras.append(f"{cap.lower()} capital")
    if key.get("age_bucket"):
        extras.append(f"age {format_comp_bucket_value('age_bucket', key.get('age_bucket'))}")
    prior = key.get("prior_finish")
    if prior:
        phrase = PRIOR_FINISH_TREND.get(str(prior), format_comp_bucket_value("prior_finish", prior).lower())
        if prior == "none":
            extras.append("with no prior season")
        else:
            extras.append(f"who finished {phrase} last year")
    if key.get("target_share"):
        extras.append(f"with {key.get('target_share')} target share last year")
    if key.get("snap_pct"):
        extras.append(f"{key.get('snap_pct')} snaps last year")
    head = f"Among {pos}s" if pos != "skill players" else "Among similar players"
    if extras:
        return f"{head} {', '.join(extras)}"
    return head


def build_hist_trends(
    query: Mapping[str, Any],
    aggregates: Mapping[str, Any],
    market: Optional[Mapping[str, Any]] = None,
) -> list[dict]:
    """Single-dimension historical slices for this player's buckets. Display only."""
    feats = extract_comp_query(query)
    pos = str(feats.get("position") or "").upper()
    if pos not in SKILL_POSITIONS:
        return []
    mkt = market if isinstance(market, Mapping) else {}
    rows: list[dict] = []

    def add(row: Optional[dict]) -> None:
        if row:
            rows.append(row)

    bucket_label = format_adp_bucket_label(mkt.get("adp_bucket"))
    add(_trend_row(
        kind="adp",
        label="ADP round",
        bucket=bucket_label,
        sentence=f"{pos}s taken in fantasy {bucket_label} finished top-12",
        rate={
            "display_pct": display_percent(mkt.get("p_top_12")),
            "sample_size": mkt.get("sample_size"),
            "confidence": mkt.get("confidence"),
        },
    ) if bucket_label else None)

    repeat = (aggregates.get("repeat_and_breakout") or {}).get(pos) or {}
    prior = feats.get("prior_finish")
    if prior in ("top_5", "top_12"):
        add(_trend_row(
            kind="repeat",
            label="Last-year elite",
            bucket="Top-12 last year",
            sentence=f"{pos}s who finished top-12 last year finished top-12 again",
            rate=repeat.get("prev_top12_to_top12"),
        ))
        add(_trend_row(
            kind="repeat_top5",
            label="Last-year elite",
            bucket="Top-12 last year",
            sentence=f"{pos}s who finished top-12 last year finished top-5 the next year",
            rate=repeat.get("prev_top12_to_top5"),
        ))
    elif prior in ("none", "top_24", "top_36", "outside_36") or not prior:
        add(_trend_row(
            kind="breakout",
            label="Breakout",
            bucket="Outside last year's top-12",
            sentence=f"{pos}s outside last year's top-12 broke into top-12",
            rate=repeat.get("engine_breakout_among_non_starters"),
        ))

    stage = feats.get("career_stage")
    if stage:
        stage_rate = (((aggregates.get("career_stages") or {}).get(pos) or {}).get("by_stage") or {}).get(stage)
        stage_label = format_comp_bucket_value("career_stage", stage)
        add(_trend_row(
            kind="career_stage",
            label="Career stage",
            bucket=stage_label,
            sentence=f"{stage_label} {pos}s finished top-12",
            rate=stage_rate,
        ))

    cap = feats.get("draft_capital")
    if cap:
        cap_rate = (
            (((aggregates.get("draft_capital") or {}).get(pos) or {}).get("season_level_by_capital") or {}).get(cap) or {}
        ).get("top_12")
        cap_label = format_comp_bucket_value("draft_capital", cap)
        add(_trend_row(
            kind="draft_capital",
            label="Draft capital",
            bucket=cap_label,
            sentence=f"NFL {cap_label} {pos}s finished top-12",
            rate=cap_rate,
        ))

    age_block = (aggregates.get("age_curves") or {}).get(pos) or {}
    age_b = feats.get("age_bucket")
    if age_b:
        age_label = format_comp_bucket_value("age_bucket", age_b)
        add(_trend_row(
            kind="age",
            label="Age",
            bucket=age_label,
            sentence=f"{pos}s age {age_label} finished top-12",
            rate=(age_block.get("by_bucket") or {}).get(age_b),
        ))
    age_int = integer_age(query.get("age"))
    if age_int is not None:
        add(_trend_row(
            kind="age_exact",
            label="Age",
            bucket=str(age_int),
            sentence=f"Age-{age_int} {pos}s finished top-12",
            rate=(age_block.get("by_integer_age") or {}).get(str(age_int)),
        ))
    prime = age_block.get("prime_window") if isinstance(age_block.get("prime_window"), Mapping) else {}
    lo, hi = prime.get("age_start"), prime.get("age_end")
    if lo is not None and hi is not None:
        pair = age_block.get("prime_window_pair") if isinstance(age_block.get("prime_window_pair"), Mapping) else {}
        add(_trend_row(
            kind="prime",
            label="Prime window",
            bucket=f"{lo}–{hi}",
            sentence=f"{pos} hit rates have peaked at ages {lo}–{hi}",
            rate=pair.get("conditional") if isinstance(pair, Mapping) else None,
        ))

    usage = aggregates.get("prior_usage") if isinstance(aggregates.get("prior_usage"), Mapping) else {}
    tgt = feats.get("target_share")
    if tgt:
        tgt_rate = (
            (((usage.get("target_share") or {}).get("by_position") or {}).get(pos) or {}).get("by_bucket") or {}
        ).get(tgt)
        add(_trend_row(
            kind="target_share",
            label="Last year target share",
            bucket=str(tgt),
            sentence=f"{pos}s with {tgt} target share last year finished top-12",
            rate=tgt_rate,
        ))
    snap = feats.get("snap_pct")
    if snap:
        snap_rate = (
            (((usage.get("snap_pct") or {}).get("by_position") or {}).get(pos) or {}).get("by_bucket") or {}
        ).get(snap)
        add(_trend_row(
            kind="snap_pct",
            label="Last year snaps",
            bucket=str(snap),
            sentence=f"{pos}s with {snap} snaps last year finished top-12",
            rate=snap_rate,
        ))
    return rows


def _info_row(
    *,
    kind: str,
    label: str,
    sentence: str,
    display: str,
    bucket: str = "",
    pct: Any = None,
    n: Any = None,
) -> dict:
    return {
        "kind": kind,
        "label": label,
        "bucket": bucket,
        "sentence": sentence,
        "display": display,
        "pct": pct,
        "n": n,
        "confidence": None,
        "confidence_label": None,
    }


def build_projection_trends(
    query: Mapping[str, Any],
    history: Optional[Mapping[str, Any]] = None,
) -> list[dict]:
    """Live board projection context. PPG is never turned into a hit rate."""
    hist = history if isinstance(history, Mapping) else {}
    pos = str(query.get("position") or "").upper() or "player"
    ppg = projected_ppg_of(query)
    rank = _optional_int(
        query.get("projected_positional_rank") or query.get("proj_rk")
    )
    adp_rk = _optional_int(
        query.get("adp_positional_rank") or query.get("adp_rk")
    )
    top12 = hist.get("rates") if isinstance(hist.get("rates"), Mapping) else {}
    top12 = top12.get("top_12") if isinstance(top12.get("top_12"), Mapping) else {}
    history_p = top12.get("smoothed_rate")
    hist_pct = top12.get("display_pct")
    implies = (rank <= 12) if rank is not None else None
    rows: list[dict] = []
    if ppg is not None:
        rows.append(_info_row(
            kind="projection_ppg",
            label="Projection",
            bucket="Sleeper PPG",
            sentence="Sleeper projection for this season",
            display=f"{ppg:.1f} PPG",
        ))
    if rank is not None:
        band = "inside the top-12" if implies else "outside the top-12"
        rows.append(_info_row(
            kind="projection_rank",
            label="Implied rank",
            bucket=band,
            sentence=f"Implied {pos} rank among this board's projections",
            display=f"#{rank}",
        ))
    vs_h = compare_projection_vs_history(implies, history_p)
    vs_h_copy = {
        "history_skeptical": (
            "Projection is a top-12; similar profiles finished top-12 less often"
        ),
        "history_bullish": (
            "Projection is outside the top-12; similar profiles finished top-12 more often"
        ),
        "agree_hit": "Projection and similar-profile history both point at a top-12",
        "agree_miss": "Projection and similar-profile history both sit outside the top-12",
    }
    sentence = vs_h_copy.get(str(vs_h.get("label") or ""))
    if sentence:
        rows.append(_info_row(
            kind="projection_vs_history",
            label="Projection vs history",
            sentence=sentence,
            display=f"{hist_pct}%" if hist_pct is not None else "—",
            pct=hist_pct,
            n=hist.get("n") or top12.get("sample_size"),
        ))
    vs_m = compare_projection_vs_market(rank, adp_rk)
    mlabel = vs_m.get("label")
    if mlabel == "projection_higher" and rank is not None and adp_rk is not None:
        rows.append(_info_row(
            kind="projection_vs_market",
            label="Projection vs ADP",
            sentence=f"Projection ranks him #{rank} vs ADP rank #{adp_rk} on this board",
            display=f"#{rank}",
            bucket=f"ADP #{adp_rk}",
        ))
    elif mlabel == "market_higher" and rank is not None and adp_rk is not None:
        rows.append(_info_row(
            kind="projection_vs_market",
            label="Projection vs ADP",
            sentence=f"ADP ranks him #{adp_rk} vs projected #{rank} on this board",
            display=f"#{adp_rk}",
            bucket=f"projected #{rank}",
        ))
    elif mlabel == "aligned" and rank is not None and adp_rk is not None:
        rows.append(_info_row(
            kind="projection_vs_market",
            label="Projection vs ADP",
            sentence=f"Projection (#{rank}) and ADP rank (#{adp_rk}) are in the same range",
            display=f"#{rank}",
            bucket=f"ADP #{adp_rk}",
        ))
    return rows


def build_hist_panel_copy(
    history: Mapping[str, Any],
    market: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Display payload for the Hist modal. No matching math."""
    hist = history if isinstance(history, Mapping) else {}
    mkt = market if isinstance(market, Mapping) else {}
    key_used = hist.get("key_used") if isinstance(hist.get("key_used"), Mapping) else {}
    rates = hist.get("rates") if isinstance(hist.get("rates"), Mapping) else {}
    hit_rates: list[dict] = []
    for tier in COMP_BOARD_TIERS:
        rec = rates.get(tier) if isinstance(rates.get(tier), Mapping) else {}
        finish = TIER_FINISH_DISPLAY.get(tier, tier.replace("_", "-"))
        label = f"Then finished {finish}"
        n = rec.get("sample_size")
        if n in (None, 0):
            n = hist.get("n")
        hit_rates.append({
            "tier": tier,
            "label": label,
            "pct": rec.get("display_pct"),
            "n": n,
            "confidence": rec.get("confidence"),
            "confidence_label": _confidence_label(rec.get("confidence")),
        })

    profile: list[dict] = []
    for dim in COMP_DIMENSION_ORDER:
        if dim not in key_used:
            continue
        value = format_comp_bucket_value(dim, key_used.get(dim))
        if not value:
            continue
        profile.append({
            "key": dim,
            "label": COMP_FEATURE_LABELS.get(dim, _title_from_key(dim)),
            "value": value,
        })

    dropped = [str(d) for d in (hist.get("dropped") or []) if d]
    relaxed = [
        {"key": dim, "label": COMP_FEATURE_LABELS.get(dim, _title_from_key(dim))}
        for dim in dropped
    ]
    relaxed_note = None
    if relaxed:
        relaxed_note = (
            "These filters were dropped so the comparison pool could reach "
            "at least 15 similar seasons."
        )

    market_sentence = format_market_sentence(mkt, missing="no_adp")
    cohort = cohort_sentence(key_used)
    cohort_note = (
        "This is a historical hit rate for that group, not this player's odds."
    )
    if relaxed_note:
        cohort_note = f"{cohort_note} {relaxed_note}"

    return {
        "headline": cohort,
        "cohort_note": cohort_note,
        "hit_rates": hit_rates,
        "profile_heading": "This pre-season profile",
        "profile": profile,
        "relaxed_heading": "Dropped to grow the sample",
        "relaxed": relaxed,
        "relaxed_note": relaxed_note,
        "trends_heading": "Trends for this player's buckets",
        "trends_note": (
            "Each row is one historical slice for a bucket this player is in. "
            "They are not combined into a ranking score."
        ),
        "trends": [],
        "projection_heading": "This board's projection",
        "projection_note": (
            "Sleeper PPG and implied rank among this cheat sheet. "
            "PPG is not turned into a hit rate."
        ),
        "projection_trends": [],
        "market_heading": "ADP bucket hit rate",
        "market_sentence": market_sentence,
        "examples_heading": "Seasons from that similar group",
        "examples_note": (
            "Notable finishes from the group above. This player is excluded. "
            "These are the hits, not a typical outcome."
        ),
    }


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
    history = {
        "n": looked.get("n"),
        "key_used": looked.get("key_used"),
        "dropped": looked.get("dropped"),
        "fallback": looked.get("fallback"),
        "rates": looked.get("rates"),
        "examples": looked.get("examples") or [],
        "kind": "conditional",
    }
    copy = build_hist_panel_copy(history, market)
    copy["trends"] = build_hist_trends(query, aggregates, market)
    copy["projection_trends"] = build_projection_trends(query, history)
    return {
        "available": True,
        "player_id": pid,
        "descriptive_only": True,
        "no_blended_score": True,
        "not_in_ranking": True,
        "preseason": extract_comp_query(query),
        "history": history,
        "market": market,
        "copy": copy,
    }
