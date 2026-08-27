"""Positional finishes and leakage-safe prior-career features (pure).

Season finish is total production unless the caller ranks a PPG column.
Ties share a competition rank (1, 2, 2, 4). Unranked players keep None
finishes rather than a last-place zero.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

from dashboard_services.historical.definitions import (
    POINTS_COLUMNS,
    PPG_COLUMNS,
    SCORING_FORMATS,
    SKILL_POSITIONS,
    TIER_CUTOFFS,
    positional_tier_label,
    tier_flags,
)

# Outcome / target columns that *may* change when a season's actuals change.
# Everything else on a feature row is pre-season and must stay put.
OUTCOME_COLUMNS = frozenset(
    {
        "games",
        "starts",
        "snaps",
        "snap_pct",
        "targets",
        "target_share",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "air_yards",
        "adot",
        "carries",
        "rush_yards",
        "rush_tds",
        "red_zone_targets",
        "red_zone_carries",
        "passing_attempts",
        "passing_yards",
        "passing_tds",
        "interceptions",
        "ppr_points",
        "half_ppr_points",
        "standard_points",
        "ppr_ppg",
        "half_ppr_ppg",
        "standard_ppg",
        "source_schema",
    }
    | {f"{fmt}_overall_finish" for fmt in SCORING_FORMATS}
    | {f"{fmt}_positional_finish" for fmt in SCORING_FORMATS}
    | {f"{fmt}_tier" for fmt in SCORING_FORMATS}
    | {f"{fmt}_{flag}" for fmt in SCORING_FORMATS for flag in TIER_CUTOFFS}
    | {f"{fmt}_ppg_overall_finish" for fmt in SCORING_FORMATS}
    | {f"{fmt}_ppg_positional_finish" for fmt in SCORING_FORMATS}
)


def _points(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def competition_ranks(values: Sequence[Optional[float]]) -> list[Optional[int]]:
    """Competition ranking (1, 2, 2, 4) over a list that may contain None.

    None values are unranked. Higher is better.
    """
    indexed = [(i, v) for i, v in enumerate(values) if v is not None]
    indexed.sort(key=lambda iv: -iv[1])
    ranks: list[Optional[int]] = [None] * len(values)
    last_val = object()
    last_rank = 0
    for place, (i, v) in enumerate(indexed, start=1):
        if v != last_val:
            last_rank = place
            last_val = v
        ranks[i] = last_rank
    return ranks


def assign_season_finishes(
    rows: Sequence[Mapping[str, Any]],
    *,
    scoring: str = "ppr",
    by_ppg: bool = False,
) -> list[dict]:
    """Copy ``rows`` and attach finish / tier columns for one scoring format.

    ``overall_finish`` is across skill positions **within each season**;
    ``positional_finish`` is within QB/RB/WR/TE in that season. Season finish
    uses total points unless ``by_ppg=True``. Rows from different seasons are
    never ranked against each other.
    """
    if scoring not in SCORING_FORMATS:
        raise ValueError(f"unknown scoring format {scoring!r}")
    points_key = PPG_COLUMNS[scoring] if by_ppg else POINTS_COLUMNS[scoring]
    prefix = f"{scoring}_ppg" if by_ppg else scoring

    out = [dict(r) for r in rows]
    by_season: dict[Any, list[int]] = {}
    for i, row in enumerate(out):
        season = row.get("season")
        by_season.setdefault(season, []).append(i)

    for idxs in by_season.values():
        points = [_points(out[i].get(points_key)) for i in idxs]
        overall = competition_ranks(points)
        for i, rank in zip(idxs, overall):
            out[i][f"{prefix}_overall_finish"] = rank

        pos_groups: dict[str, list[int]] = {p: [] for p in SKILL_POSITIONS}
        for j, i in enumerate(idxs):
            pos = str(out[i].get("position") or "").upper()
            if pos in pos_groups:
                pos_groups[pos].append(j)
        for pos, local in pos_groups.items():
            pos_points = [points[j] for j in local]
            pos_ranks = competition_ranks(pos_points)
            for j, rank in zip(local, pos_ranks):
                i = idxs[j]
                row = out[i]
                row[f"{prefix}_positional_finish"] = rank
                if not by_ppg:
                    row[f"{prefix}_tier"] = positional_tier_label(pos, rank)
                    for flag, value in tier_flags(rank).items():
                        row[f"{prefix}_{flag}"] = value

    for row in out:
        row.setdefault(f"{prefix}_positional_finish", None)
        if not by_ppg:
            row.setdefault(f"{prefix}_tier", None)
            for flag in TIER_CUTOFFS:
                row.setdefault(f"{prefix}_{flag}", False)
    return out


def assign_all_scoring_finishes(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Attach PPR, half-PPR, and standard total-points finishes (and PPR PPG)."""
    out = [dict(r) for r in rows]
    for scoring in SCORING_FORMATS:
        out = assign_season_finishes(out, scoring=scoring, by_ppg=False)
    out = assign_season_finishes(out, scoring="ppr", by_ppg=True)
    return out


def _prior_stats(prior: Sequence[Mapping[str, Any]], scoring: str) -> dict:
    finish_key = f"{scoring}_positional_finish"
    ppg_key = PPG_COLUMNS[scoring]
    finishes = []
    ppgs = []
    games_list = []
    for row in prior:
        fin = row.get(finish_key)
        if fin is not None:
            try:
                finishes.append(int(fin))
            except (TypeError, ValueError):
                pass
        ppg = _points(row.get(ppg_key))
        if ppg is not None:
            ppgs.append(ppg)
        g = row.get("games")
        if g is not None:
            try:
                games_list.append(int(g))
            except (TypeError, ValueError):
                pass

    counts = {}
    flags = {}
    for name, cutoff in TIER_CUTOFFS.items():
        n = sum(1 for f in finishes if f <= cutoff)
        counts[name] = n
        flags[name] = n > 0

    prev = prior[-1] if prior else None
    previous_finish = None
    previous_ppg = None
    previous_games = None
    if prev is not None:
        try:
            previous_finish = int(prev[finish_key]) if prev.get(finish_key) is not None else None
        except (TypeError, ValueError, KeyError):
            previous_finish = None
        previous_ppg = _points(prev.get(ppg_key))
        try:
            previous_games = int(prev["games"]) if prev.get("games") is not None else None
        except (TypeError, ValueError, KeyError):
            previous_games = None

    career_best_finish = min(finishes) if finishes else None
    career_best_ppg = max(ppgs) if ppgs else None
    top12 = flags.get("top_12", False)
    return {
        "previous_season_finish": previous_finish,
        "previous_season_ppg": previous_ppg,
        "previous_season_games": previous_games,
        "career_best_finish_before_season": career_best_finish,
        "career_best_ppg_before_season": career_best_ppg,
        "prior_top3_count": counts.get("top_3", 0),
        "prior_top5_count": counts.get("top_5", 0),
        "prior_top6_count": counts.get("top_6", 0),
        "prior_top12_count": counts.get("top_12", 0),
        "prior_top24_count": counts.get("top_24", 0),
        "previously_top3": flags.get("top_3", False),
        "previously_top5": flags.get("top_5", False),
        "previously_top6": flags.get("top_6", False),
        "previously_top12": flags.get("top_12", False),
        "previously_top24": flags.get("top_24", False),
        "first_time_top12_candidate": not top12,
        "career_seasons_before_current": len(prior),
    }


def prior_career_features_for_player(
    seasons: Sequence[Mapping[str, Any]],
    *,
    scoring: str = "ppr",
) -> list[dict]:
    """One feature row per player-season using only seasons *before* that year.

    ``seasons`` is this player's career, any order. Same-season actuals never
    enter the feature columns. Outcome columns on the returned rows are copies
    of that season's already-computed finishes/points (targets), not features.
    """
    if scoring not in SCORING_FORMATS:
        raise ValueError(f"unknown scoring format {scoring!r}")
    ordered = sorted(seasons, key=lambda r: int(r.get("season") or 0))
    out = []
    for i, row in enumerate(ordered):
        prior = [r for r in ordered[:i] if int(r.get("season") or 0) < int(row.get("season") or 0)]
        stats = _prior_stats(prior, scoring)
        feature_row = dict(row)
        prefix = "" if scoring == "ppr" else f"{scoring}_"
        # Primary (PPR) columns keep the spec names; other formats are prefixed
        # so a single row can carry all three without clobbering.
        if scoring == "ppr":
            feature_row.update(stats)
        else:
            for key, value in stats.items():
                feature_row[f"{prefix}{key}"] = value
        out.append(feature_row)
    return out


def attach_prior_career_features(
    rows: Iterable[Mapping[str, Any]],
    *,
    scoring_formats: Sequence[str] = SCORING_FORMATS,
) -> list[dict]:
    """Group by sleeper_id and attach leakage-safe prior-career features."""
    by_player: dict[str, list] = {}
    for row in rows:
        pid = str(row.get("sleeper_id") or row.get("player_id") or "")
        if not pid:
            continue
        by_player.setdefault(pid, []).append(dict(row))
    attached: list[dict] = []
    for pid, career in by_player.items():
        framed = [dict(r) for r in career]
        for scoring in scoring_formats:
            framed = prior_career_features_for_player(framed, scoring=scoring)
        attached.extend(framed)
    attached.sort(key=lambda r: (int(r.get("season") or 0), str(r.get("sleeper_id") or "")))
    return attached
