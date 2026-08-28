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
    ABSOLUTE_BUST_OUTSIDE,
    ADP_OVERALL_BUCKETS,
    ADOT_BUCKETS,
    AGE_BUCKETS,
    CAREER_STAGE_ORDER,
    COMP_BOARD_TIERS,
    COMP_DIMENSION_ORDER,
    DRAFT_CAPITAL_ORDER,
    PRIOR_FINISH_BUCKETS,
    RYOE_BUCKETS,
    SKILL_POSITIONS,
    SNAP_PCT_BUCKETS,
    TARGET_SHARE_BUCKETS,
    display_percent,
    draft_capital_bucket,
    integer_age,
    normalize_adp,
    value_bucket,
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
    "previous_season_adot",
    "previous_season_ngs_rush_yards_over_expected_per_att",
    "previous_season_year",
    "prior_top12_count",
)


def _upcoming_top12_count(row: Mapping[str, Any]) -> Optional[int]:
    """Top-12 seasons through the last warehouse year, for the upcoming preseason."""
    prev = _optional_int(row.get("prior_top12_count"))
    last_finish = _optional_int(row.get("ppr_positional_finish"))
    if last_finish is not None and last_finish <= 12:
        return (prev or 0) + 1
    return prev

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
    "day_2": "Day 2 (rounds 2-3)",
    "day_3": "Day 3 (rounds 4-7)",
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
    "rounds_6_7": "Rounds 6-7",
    "rounds_8_10": "Rounds 8-10",
    "rounds_11_plus": "Rounds 11+",
}
ADP_POSITIONAL_DISPLAY: dict[str, str] = {
    "top_5": "Positional ADP 1-5",
    "top_12": "Positional ADP 6-12",
    "top_24": "Positional ADP 13-24",
    "top_36": "Positional ADP 25-36",
    "outside_36": "Positional ADP 37+",
}
CUMULATIVE_TREND_WINDOWS: tuple[tuple[str, str, str], ...] = (
    (
        "top12_as_rookie",
        "Hit top-12 as a rookie",
        "Share of that NFL draft class who posted a top-12 season in year 1. Player-level, not a single-season rate.",
    ),
    (
        "top12_by_year_2",
        "Hit top-12 by year 2",
        "Share who posted a top-12 season in year 1 or year 2.",
    ),
)
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
        "trends_tab": "/api/historical-trends",
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
            "previous_season_adot": _optional_float(row.get("adot")),
            "previous_season_ngs_rush_yards_over_expected_per_att": _optional_float(
                row.get("ngs_rush_yards_over_expected_per_att")
            ),
            "previous_season_year": last_season,
            "prior_top12_count": _upcoming_top12_count(row),
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
        if val is None:
            val = player.get(key)
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
    return text


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


def format_adot_bucket_label(bucket: Any) -> str:
    text = str(bucket or "").strip()
    if not text:
        return ""
    return f"aDOT {text} yards"


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
    baseline_pct: Any = None,
    secondary: Optional[str] = None,
    polarity: Optional[str] = None,
) -> Optional[dict]:
    rec = _as_rate(rate)
    pct = rec.get("display_pct")
    if pct is None:
        return None
    sample = rec.get("sample_size")
    if sample is None:
        sample = rec.get("n_players")
    row: dict[str, Any] = {
        "kind": kind,
        "label": label,
        "bucket": bucket,
        "sentence": sentence,
        "pct": pct,
        "n": sample,
        "confidence": rec.get("confidence"),
        "confidence_label": _confidence_label(rec.get("confidence")),
    }
    if secondary:
        row["secondary"] = secondary
    if polarity:
        row["polarity"] = polarity
    if isinstance(baseline_pct, (int, float)):
        delta = int(pct) - int(baseline_pct)
        row["vs_baseline"] = delta
        if delta > 0:
            row["vs_label"] = f"+{delta} vs typical"
        elif delta < 0:
            row["vs_label"] = f"{delta} vs typical"
        else:
            row["vs_label"] = "in line with typical"
    return row


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


def _position_baseline_pct(aggregates: Mapping[str, Any], pos: str) -> Any:
    age_block = (aggregates.get("age_curves") or {}).get(pos) or {}
    baseline = age_block.get("baseline") if isinstance(age_block.get("baseline"), Mapping) else {}
    if not baseline:
        baseline = ((aggregates.get("career_stages") or {}).get(pos) or {}).get("baseline") or {}
    return baseline.get("display_pct") if isinstance(baseline, Mapping) else None


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
    baseline_pct = _position_baseline_pct(aggregates, pos)
    rows: list[dict] = []

    def add(row: Optional[dict]) -> None:
        if row:
            rows.append(row)

    repeat = (aggregates.get("repeat_and_breakout") or {}).get(pos) or {}
    prior = feats.get("prior_finish")
    if prior in ("top_5", "top_12"):
        add(_trend_row(
            kind="repeat",
            label="Last-year elite",
            bucket="Top-12 last year",
            sentence=f"{pos}s who finished top-12 last year finished top-12 again",
            rate=repeat.get("prev_top12_to_top12"),
            baseline_pct=baseline_pct,
        ))
        add(_trend_row(
            kind="repeat_top5",
            label="Last-year elite",
            bucket="Top-12 last year",
            sentence=f"{pos}s who finished top-12 last year finished top-5 the next year",
            rate=repeat.get("prev_top12_to_top5"),
        ))
        prior_count = _optional_int(query.get("prior_top12_count"))
        if prior_count is not None and prior_count >= 2:
            add(_trend_row(
                kind="two_plus",
                label="Repeat stars",
                bucket="Two or more prior top-12s",
                sentence=f"{pos}s with two or more prior top-12 seasons finished top-12 again",
                rate=repeat.get("two_plus_prior_top12_to_top12"),
                baseline_pct=baseline_pct,
            ))
    elif prior in ("none", "top_24", "top_36", "outside_36") or not prior:
        add(_trend_row(
            kind="breakout",
            label="Breakout",
            bucket="Outside last year's top-12",
            sentence=f"{pos}s outside last year's top-12 broke into top-12",
            rate=repeat.get("engine_breakout_among_non_starters"),
            baseline_pct=baseline_pct,
        ))
        add(_trend_row(
            kind="first_time_elite",
            label="First-time elite",
            bucket="Never previously top-12",
            sentence=f"{pos}s who had never been top-12 broke into top-12",
            rate=repeat.get("first_time_elite_among_candidates"),
            baseline_pct=baseline_pct,
        ))
        add(_trend_row(
            kind="league_winner_smash",
            label="League-winner smash",
            bucket="Outside last year's top-12",
            sentence=f"{pos}s outside last year's top-12 finished top-5",
            rate=repeat.get("league_winner_smash_among_non_top12"),
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
            baseline_pct=baseline_pct,
        ))

    capital = (aggregates.get("draft_capital") or {}).get(pos) or {}
    cap = feats.get("draft_capital")
    cap_label = format_comp_bucket_value("draft_capital", cap) if cap else ""
    cap_rec = ((capital.get("season_level_by_capital") or {}).get(cap) or {}) if cap else {}
    if cap:
        top5 = _as_rate(cap_rec.get("top_5")).get("display_pct")
        add(_trend_row(
            kind="draft_capital",
            label="Draft capital",
            bucket=cap_label,
            sentence=f"NFL {cap_label} {pos}s finished top-12",
            rate=cap_rec.get("top_12"),
            baseline_pct=baseline_pct,
            secondary=f"{top5}% top-5" if top5 is not None else None,
        ))
        bust_cut = ABSOLUTE_BUST_OUTSIDE.get(pos)
        if bust_cut is not None:
            add(_trend_row(
                kind="capital_miss",
                label="Miss rate",
                bucket=cap_label,
                sentence=f"NFL {cap_label} {pos}s finished outside the top-{bust_cut}",
                rate=cap_rec.get("absolute_bust"),
                polarity="miss",
            ))
        for window_id, heading, _note in CUMULATIVE_TREND_WINDOWS:
            window = (capital.get("cumulative") or {}).get(window_id) or {}
            when = "as a rookie" if window_id == "top12_as_rookie" else "by year 2"
            add(_trend_row(
                kind=window_id,
                label=heading,
                bucket=cap_label,
                sentence=f"NFL {cap_label} {pos}s posted a top-12 {when}",
                rate=(window.get("by_capital") or {}).get(cap),
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
            baseline_pct=baseline_pct,
        ))
    age_int = integer_age(query.get("age"))
    if age_int is not None:
        add(_trend_row(
            kind="age_exact",
            label="Age",
            bucket=str(age_int),
            sentence=f"Age-{age_int} {pos}s finished top-12",
            rate=(age_block.get("by_integer_age") or {}).get(str(age_int)),
            baseline_pct=baseline_pct,
        ))
    prime = age_block.get("prime_window") if isinstance(age_block.get("prime_window"), Mapping) else {}
    lo, hi = prime.get("age_start"), prime.get("age_end")
    if lo is not None and hi is not None:
        pair = age_block.get("prime_window_pair") if isinstance(age_block.get("prime_window_pair"), Mapping) else {}
        add(_trend_row(
            kind="prime",
            label="Prime window",
            bucket=f"{lo}-{hi}",
            sentence=f"{pos} hit rates have peaked at ages {lo}-{hi}",
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
            baseline_pct=baseline_pct,
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
            baseline_pct=baseline_pct,
        ))
    adot = value_bucket(query.get("previous_season_adot"), ADOT_BUCKETS)
    if adot:
        adot_rate = (
            (((usage.get("adot") or {}).get("by_position") or {}).get(pos) or {}).get("by_bucket") or {}
        ).get(adot)
        add(_trend_row(
            kind="adot",
            label="Last year aDOT",
            bucket=format_adot_bucket_label(adot),
            sentence=f"{pos}s with {format_adot_bucket_label(adot)} last year finished top-12",
            rate=adot_rate,
            baseline_pct=baseline_pct,
        ))
    ryoe = value_bucket(
        query.get("previous_season_ngs_rush_yards_over_expected_per_att"),
        RYOE_BUCKETS,
    )
    if ryoe:
        ryoe_rate = (
            (((usage.get("ryoe") or {}).get("by_position") or {}).get(pos) or {}).get("by_bucket") or {}
        ).get(ryoe)
        add(_trend_row(
            kind="ryoe",
            label="Last year RYOE",
            bucket=str(ryoe),
            sentence=f"{pos}s with {ryoe} last-year RYOE finished top-12",
            rate=ryoe_rate,
            baseline_pct=baseline_pct,
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
            "Each row is one historical slice for a bucket this player is in, "
            "including ADP rank, NFL capital, miss rates, and usage. "
            "+N vs typical is versus a typical player-season at the position. "
            "They are not combined into a ranking score."
        ),
        "trends": [],
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


def _era_label(aggregates: Mapping[str, Any]) -> str:
    rng = aggregates.get("season_range") or []
    if isinstance(rng, (list, tuple)) and len(rng) >= 2 and rng[0] is not None and rng[1] is not None:
        return f"{rng[0]}-{rng[1]}"
    return "2018-2025"


def _section_row(
    label: str,
    rate: Any,
    *,
    baseline_pct: Any = None,
    secondary: Optional[str] = None,
) -> Optional[dict]:
    rec = _as_rate(rate)
    pct = rec.get("display_pct")
    if pct is None:
        return None
    sample = rec.get("sample_size")
    if sample is None:
        sample = rec.get("n_players")
    row: dict[str, Any] = {
        "label": label,
        "pct": pct,
        "n": sample,
        "confidence": rec.get("confidence"),
        "confidence_label": _confidence_label(rec.get("confidence")),
    }
    if secondary:
        row["secondary"] = secondary
    if isinstance(baseline_pct, (int, float)):
        delta = int(pct) - int(baseline_pct)
        row["vs_baseline"] = delta
        if delta > 0:
            row["vs_label"] = f"+{delta} vs typical"
        elif delta < 0:
            row["vs_label"] = f"{delta} vs typical"
        else:
            row["vs_label"] = "in line with typical"
    return row


def _append_section(
    sections: list[dict],
    *,
    sid: str,
    heading: str,
    note: str,
    rows: list[dict],
    polarity: Optional[str] = None,
) -> None:
    if not rows:
        return
    rec: dict[str, Any] = {"id": sid, "heading": heading, "note": note, "rows": rows}
    if polarity:
        rec["polarity"] = polarity
    sections.append(rec)


def _trend_highlights(sections: Sequence[Mapping[str, Any]], *, limit: int = 4) -> list[dict]:
    """Biggest above-typical buckets, preferring one from each table first."""
    best_by_section: list[dict] = []
    leftovers: list[dict] = []
    for sec in sections:
        if str(sec.get("polarity") or "") == "miss":
            continue
        scored = []
        for row in sec.get("rows") or []:
            vs = row.get("vs_baseline")
            if not isinstance(vs, int) or vs <= 0:
                continue
            scored.append({
                "section": sec.get("heading"),
                "label": row.get("label"),
                "pct": row.get("pct"),
                "vs_baseline": vs,
                "vs_label": row.get("vs_label"),
                "n": row.get("n"),
            })
        scored.sort(key=lambda r: (-int(r.get("vs_baseline") or 0), -int(r.get("pct") or 0)))
        if scored:
            best_by_section.append(scored[0])
            leftovers.extend(scored[1:])
    best_by_section.sort(key=lambda r: (-int(r.get("vs_baseline") or 0), -int(r.get("pct") or 0)))
    leftovers.sort(key=lambda r: (-int(r.get("vs_baseline") or 0), -int(r.get("pct") or 0)))
    picked = best_by_section[:limit]
    if len(picked) < limit:
        picked.extend(leftovers[: limit - len(picked)])
    return picked[:limit]


def _age_curve_points(age_block: Mapping[str, Any]) -> list[dict]:
    by_int = age_block.get("by_integer_age") if isinstance(age_block.get("by_integer_age"), Mapping) else {}
    points: list[dict] = []
    for key in sorted(by_int.keys(), key=lambda k: int(k) if str(k).isdigit() else 99):
        if not str(key).isdigit():
            continue
        rec = _as_rate(by_int.get(key))
        pct = rec.get("display_pct")
        if pct is None:
            continue
        points.append({
            "age": int(key),
            "pct": pct,
            "n": rec.get("sample_size"),
        })
    return points


def build_position_trend_page(aggregates: Mapping[str, Any], position: str) -> dict:
    """Display tables for one position. JSON lookup only."""
    pos = str(position or "").upper()
    sections: list[dict] = []
    age_block = (aggregates.get("age_curves") or {}).get(pos) or {}
    baseline = age_block.get("baseline") if isinstance(age_block.get("baseline"), Mapping) else {}
    if not baseline:
        baseline = ((aggregates.get("career_stages") or {}).get(pos) or {}).get("baseline") or {}
    baseline_pct = baseline.get("display_pct") if isinstance(baseline, Mapping) else None
    prime = age_block.get("prime_window") if isinstance(age_block.get("prime_window"), Mapping) else {}
    lo, hi = prime.get("age_start"), prime.get("age_end")
    prime_label = f"{lo}-{hi}" if lo is not None and hi is not None else ""
    prime_ages = prime.get("ages") if isinstance(prime.get("ages"), list) else []
    if not prime_ages and lo is not None and hi is not None:
        try:
            prime_ages = list(range(int(lo), int(hi) + 1))
        except (TypeError, ValueError):
            prime_ages = []

    repeat = (aggregates.get("repeat_and_breakout") or {}).get(pos) or {}
    repeat_rows = []
    for label, key in (
        ("Last-year top-12 finished top-12 again", "prev_top12_to_top12"),
        ("Last-year top-12 finished top-5 next", "prev_top12_to_top5"),
        ("Two-time top-12 finished top-12 again", "two_plus_prior_top12_to_top12"),
        ("Outside last-year top-12 broke into top-12", "engine_breakout_among_non_starters"),
        ("Never-elite broke into top-12", "first_time_elite_among_candidates"),
    ):
        row = _section_row(label, repeat.get(key), baseline_pct=baseline_pct)
        if row:
            repeat_rows.append(row)
    _append_section(
        sections,
        sid="repeat",
        heading="Repeat and breakout",
        note=f"What {pos}s did the year after an elite or non-elite finish.",
        rows=repeat_rows,
    )

    winner_rows = []
    for label, key in (
        ("Finished top-5 that season", "league_winner"),
        ("Outside last-year top-12 then top-5", "league_winner_smash_among_non_top12"),
    ):
        row = _section_row(label, repeat.get(key))
        if row:
            winner_rows.append(row)
    _append_section(
        sections,
        sid="league_winner",
        heading="League winners",
        note=f"Top-5 finishes for {pos}s. Smash is a top-5 from outside last year's top-12.",
        rows=winner_rows,
    )

    stage_map = ((aggregates.get("career_stages") or {}).get(pos) or {}).get("by_stage") or {}
    stage_rows = []
    for key in CAREER_STAGE_ORDER:
        row = _section_row(
            CAREER_STAGE_DISPLAY.get(key, _title_from_key(key)),
            stage_map.get(key),
            baseline_pct=baseline_pct,
        )
        if row:
            stage_rows.append(row)
    _append_section(
        sections,
        sid="career_stage",
        heading="Career stage",
        note=f"Top-12 rate for {pos}s at that point in a career.",
        rows=stage_rows,
    )

    capital = (aggregates.get("draft_capital") or {}).get(pos) or {}
    cap_map = capital.get("season_level_by_capital") or {}
    cap_rows = []
    for key in DRAFT_CAPITAL_ORDER:
        rec = (cap_map.get(key) or {}).get("top_12")
        top5 = _as_rate((cap_map.get(key) or {}).get("top_5")).get("display_pct")
        secondary = f"{top5}% top-5" if top5 is not None else None
        row = _section_row(
            DRAFT_CAPITAL_DISPLAY.get(key, _title_from_key(key)),
            rec,
            baseline_pct=baseline_pct,
            secondary=secondary,
        )
        if row:
            cap_rows.append(row)
    _append_section(
        sections,
        sid="draft_capital",
        heading="NFL draft capital",
        note=f"Season-level top-12 rate for {pos}s by NFL draft capital, not fantasy ADP.",
        rows=cap_rows,
    )

    for window_id, heading, note in CUMULATIVE_TREND_WINDOWS:
        window = (capital.get("cumulative") or {}).get(window_id) or {}
        by_cap = window.get("by_capital") or {}
        rows = []
        for key in DRAFT_CAPITAL_ORDER:
            row = _section_row(
                DRAFT_CAPITAL_DISPLAY.get(key, _title_from_key(key)),
                by_cap.get(key),
            )
            if row:
                rows.append(row)
        _append_section(
            sections,
            sid=window_id,
            heading=heading,
            note=f"{note} NFL draft capital, not fantasy ADP.",
            rows=rows,
        )

    bust_cut = ABSOLUTE_BUST_OUTSIDE.get(pos)
    bust_rows = []
    if bust_cut is not None:
        for key in DRAFT_CAPITAL_ORDER:
            row = _section_row(
                DRAFT_CAPITAL_DISPLAY.get(key, _title_from_key(key)),
                (cap_map.get(key) or {}).get("absolute_bust"),
            )
            if row:
                bust_rows.append(row)
        _append_section(
            sections,
            sid="capital_miss",
            heading="Miss rates by NFL capital",
            note=(
                f"Share of {pos}s who finished outside the top-{bust_cut}. "
                "Higher is a miss, not a hit."
            ),
            rows=bust_rows,
            polarity="miss",
        )

    age_rows = []
    for _lo, _hi, key in AGE_BUCKETS.get(pos, ()):
        row = _section_row(
            format_age_bucket_label(key),
            (age_block.get("by_bucket") or {}).get(key),
            baseline_pct=baseline_pct,
        )
        if row:
            age_rows.append(row)
    _append_section(
        sections,
        sid="age",
        heading="Age",
        note=f"Top-12 rate for {pos}s in that age bucket.",
        rows=age_rows,
    )

    usage = aggregates.get("prior_usage") if isinstance(aggregates.get("prior_usage"), Mapping) else {}
    tgt_map = (((usage.get("target_share") or {}).get("by_position") or {}).get(pos) or {}).get("by_bucket") or {}
    tgt_rows = []
    for _lo, _hi, key in TARGET_SHARE_BUCKETS:
        row = _section_row(
            str(key),
            tgt_map.get(key),
            baseline_pct=baseline_pct,
        )
        if row:
            tgt_rows.append(row)
    _append_section(
        sections,
        sid="target_share",
        heading="Last year target share",
        note=f"How often {pos}s with that prior-season target share finished top-12.",
        rows=tgt_rows,
    )

    snap_map = (((usage.get("snap_pct") or {}).get("by_position") or {}).get(pos) or {}).get("by_bucket") or {}
    snap_rows = []
    for _lo, _hi, key in SNAP_PCT_BUCKETS:
        row = _section_row(
            str(key),
            snap_map.get(key),
            baseline_pct=baseline_pct,
        )
        if row:
            snap_rows.append(row)
    _append_section(
        sections,
        sid="snap_pct",
        heading="Last year snap share",
        note=f"How often {pos}s with that prior-season snap share finished top-12.",
        rows=snap_rows,
    )

    adot_map = (((usage.get("adot") or {}).get("by_position") or {}).get(pos) or {}).get("by_bucket") or {}
    adot_rows = []
    for _lo, _hi, key in ADOT_BUCKETS:
        row = _section_row(
            format_adot_bucket_label(key),
            adot_map.get(key),
            baseline_pct=baseline_pct,
        )
        if row:
            adot_rows.append(row)
    _append_section(
        sections,
        sid="adot",
        heading="Last year aDOT",
        note=f"How often {pos}s with that prior-season average depth of target finished top-12.",
        rows=adot_rows,
    )

    ryoe_map = (((usage.get("ryoe") or {}).get("by_position") or {}).get(pos) or {}).get("by_bucket") or {}
    ryoe_rows = []
    for _lo, _hi, key in RYOE_BUCKETS:
        row = _section_row(str(key), ryoe_map.get(key), baseline_pct=baseline_pct)
        if row:
            ryoe_rows.append(row)
    _append_section(
        sections,
        sid="ryoe",
        heading="Last year rush yards over expected",
        note=f"How often {pos}s with that prior-season RYOE finished top-12.",
        rows=ryoe_rows,
    )

    return {
        "position": pos,
        "baseline_pct": baseline_pct if isinstance(baseline, Mapping) else None,
        "baseline_n": baseline.get("sample_size") if isinstance(baseline, Mapping) else None,
        "prime_window": prime_label,
        "prime_ages": prime_ages,
        "age_curve": _age_curve_points(age_block),
        "highlights": _trend_highlights(sections),
        "sections": sections,
    }


def build_historical_trends(aggregates: Mapping[str, Any]) -> dict:
    """Position-level trend tables for the cheat-sheet Trends tab."""
    if not aggregates:
        return {"available": False, "descriptive_only": True, "not_in_ranking": True}
    era = _era_label(aggregates)
    by_pos = {pos: build_position_trend_page(aggregates, pos) for pos in SKILL_POSITIONS}
    return {
        "available": True,
        "descriptive_only": True,
        "not_in_ranking": True,
        "not_in_pick_score": True,
        "era": era,
        "headline": "Historical top-12 rates by bucket. Not a ranking score.",
        "note": (
            f"Each table is one slice from {era}. Callouts are the biggest edges "
            "versus a typical player-season at this position. Open a player's Hist "
            "button on the Big Board for that player's own mix of buckets."
        ),
        "positions": list(SKILL_POSITIONS),
        "by_position": by_pos,
    }
