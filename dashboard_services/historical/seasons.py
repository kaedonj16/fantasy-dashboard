"""Canonicalize heterogeneous usage_rows into one player-season dict.

Committed ``cache/player_history/usage_rows_{season}.json`` files come in two
shapes:

* 2018–2022 (nflverse/breakout-engine): season *totals* under ``usage``
  (``targets``, ``ppr_total``, ``gsis_id``, ``snap_share``).
* 2023–present (``build_usage_rows_for_season``): per-game *averages*
  (``avg_targets``, ``ppr_ppg``) plus Footballguys target share.

This module is pure: identity lookups are passed in. Missing values are
None, never a meaningful-looking zero.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from dashboard_services.historical.definitions import (
    SKILL_POSITIONS,
    EFFICIENCY_FIELDS,
    age_as_of_season_start,
    age_bucket,
    draft_capital_bucket,
    years_experience_before_season,
)

# Canonical column order for the player-season warehouse (Phase 1).
CANONICAL_SEASON_COLUMNS = (
    "season",
    "player_id",
    "sleeper_id",
    "gsis_id",
    "name",
    "position",
    "team",
    "age",
    "age_bucket",
    "years_experience",
    "draft_year",
    "nfl_draft_round",
    "nfl_draft_pick",
    "draft_capital_bucket",
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
    # Coverage / provenance (not a fantasy stat).
    "source_schema",
)

# Filled by Phase 3 overlay when cache is present. Canonicalize emits None.
EFFICIENCY_COVERAGE_COLUMNS = tuple(
    dict.fromkeys(CANONICAL_SEASON_COLUMNS + EFFICIENCY_FIELDS)
)

def _f(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def _i(value: Any) -> Optional[int]:
    number = _f(value)
    if number is None:
        return None
    return int(number)


def _str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "null", "unknown"):
        return None
    return text


def _product(per_game: Any, games: Any) -> Optional[float]:
    rate = _f(per_game)
    g = _f(games)
    if rate is None or g is None or g <= 0:
        return None
    return rate * g


def _prefer(*values: Any) -> Optional[float]:
    for value in values:
        number = _f(value)
        if number is not None:
            return number
    return None


def _snap_pct(raw: Any, games: Optional[float], touches: Optional[float]) -> Optional[float]:
    """Treat a 0% snap rate with real usage as missing, not a real zero.

    Legacy 2018–2022 rows often store ``snap_share: 0.0`` when nflverse snaps
    were unavailable (e.g. Tom Brady 2018). A starter with 100+ touches at 0%
    snaps is not a real observation.
    """
    pct = _f(raw)
    if pct is None:
        return None
    if pct > 1.5:  # stored as 0–100
        pct = pct / 100.0
    g = games or 0.0
    t = touches or 0.0
    if pct == 0.0 and g >= 1 and t > 20:
        return None
    if pct < 0:
        return None
    return pct


def _target_share(
    raw_share: Any,
    targets: Optional[float],
    games: Optional[float],
) -> Optional[float]:
    """0-with-targets is missing Footballguys/PBP data, not a real 0 share."""
    share = _f(raw_share)
    if share is None:
        return None
    if share < 0:
        return None
    if share == 0.0 and (targets or 0) > 0:
        return None
    if share == 0.0 and (games or 0) <= 0:
        return None
    return share


def canonicalize_usage_row(
    raw: Mapping[str, Any],
    season: int,
    identity: Optional[Mapping[str, Any]] = None,
) -> Optional[dict]:
    """Turn one usage_rows entry into a canonical player-season dict.

    Returns None when the row cannot be identified as a skill-position
    player (no sleeper id, or position outside QB/RB/WR/TE after identity
    join). Kickers/IDP are dropped — they are not part of this warehouse.
    """
    identity = identity or {}
    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    sleeper_id = _str(raw.get("sleeper_id") or raw.get("player_id") or raw.get("id"))
    if sleeper_id is None:
        return None

    position = _str(
        raw.get("position")
        or usage.get("position")
        or identity.get("position")
    )
    if position:
        position = position.upper()
        if position == "PK":
            position = "K"
    if position not in SKILL_POSITIONS:
        return None

    name = _str(raw.get("name") or raw.get("player_name") or usage.get("name") or identity.get("name"))
    team = _str(raw.get("team") or usage.get("team") or identity.get("team"))
    gsis_id = _str(usage.get("gsis_id") or raw.get("gsis_id") or identity.get("gsis_id"))

    games = _prefer(usage.get("games"), raw.get("games"))
    if games is not None and games < 0:
        games = None

    # Totals: prefer explicit season totals (legacy schema), else avg * games.
    targets = _prefer(usage.get("targets"), _product(usage.get("avg_targets"), games))
    receptions = _prefer(usage.get("receptions"), _product(usage.get("avg_receptions"), games))
    receiving_yards = _prefer(
        usage.get("rec_yards"),
        usage.get("receiving_yards"),
        _product(usage.get("avg_rec_yards"), games),
    )
    receiving_tds = _prefer(
        usage.get("rec_tds"),
        usage.get("receiving_tds"),
        _product(usage.get("avg_rec_tds"), games),
    )
    carries = _prefer(usage.get("carries"), _product(usage.get("avg_carries"), games))
    rush_yards = _prefer(
        usage.get("rush_yards"),
        usage.get("rushing_yards"),
        _product(usage.get("avg_rush_yards"), games),
    )
    rush_tds = _prefer(
        usage.get("rush_tds"),
        usage.get("rushing_tds"),
        _product(usage.get("avg_rush_tds"), games),
    )
    passing_attempts = _prefer(
        usage.get("pass_attempts"),
        usage.get("passing_attempts"),
        usage.get("attempts"),
        _product(usage.get("avg_pass_att"), games),
    )
    passing_yards = _prefer(
        usage.get("pass_yards"),
        usage.get("passing_yards"),
        _product(usage.get("avg_pass_yds"), games),
    )
    passing_tds = _prefer(
        usage.get("pass_tds"),
        usage.get("passing_tds"),
        _product(usage.get("avg_pass_tds"), games),
    )
    interceptions = _prefer(
        usage.get("interceptions"),
        usage.get("pass_int"),
        _product(usage.get("avg_pass_int"), games),
    )

    ppr_ppg = _f(usage.get("ppr_ppg"))
    half_ppg = _f(usage.get("half_ppr_ppg"))
    std_ppg = _prefer(usage.get("std_scoring_ppg"), usage.get("std_ppg") if _f(usage.get("std_ppg")) else None)
    # ``std_ppg`` on modern rows is often the leftover 0.0 alias; ignore a 0
    # when std_scoring_ppg is also missing and we can derive from PPR.
    if std_ppg == 0.0 and half_ppg is None and ppr_ppg is not None:
        std_ppg = None

    ppr_points = _prefer(usage.get("ppr_total"), usage.get("ppr_points"), _product(ppr_ppg, games))
    half_points = _prefer(usage.get("half_ppr_total"), usage.get("half_ppr_points"), _product(half_ppg, games))
    std_points = _prefer(
        usage.get("std_total"),
        usage.get("standard_points"),
        _product(std_ppg, games),
    )

    # Derive half/standard from PPR + receptions when the source only stored PPR
    # (legacy 2018–2022). This is a linear scoring identity, not invented stats.
    if ppr_points is not None and receptions is not None:
        if half_points is None:
            half_points = ppr_points - 0.5 * receptions
        if std_points is None:
            std_points = ppr_points - receptions

    if games and games > 0:
        if ppr_ppg is None and ppr_points is not None:
            ppr_ppg = ppr_points / games
        if half_ppg is None and half_points is not None:
            half_ppg = half_points / games
        if std_ppg is None and std_points is not None:
            std_ppg = std_points / games

    # Empty usage with no games and no points: keep the row only if identity
    # filled a skill position (injured / unsigned season). Stats stay None.
    played = (games or 0) > 0 or (ppr_points or 0) > 0 or (targets or 0) > 0 or (carries or 0) > 0
    if not played:
        games = games if games else None

    rz_targets = _prefer(
        usage.get("red_zone_targets"),
        _product(usage.get("rec_rz_tgt_pg"), games),
    )
    rz_carries = _prefer(
        usage.get("red_zone_carries"),
        _product(usage.get("rush_rz_att_pg"), games),
    )

    snaps = _prefer(
        usage.get("snaps"),
        usage.get("off_snaps"),
        usage.get("total_off_snaps"),
        _product(usage.get("avg_off_snaps"), games),
    )
    if snaps == 0.0:
        snaps = None
    touches = None
    parts = [targets, carries, passing_attempts]
    if any(p is not None for p in parts):
        touches = sum(p or 0 for p in parts)
    snap_pct = _snap_pct(
        usage.get("snap_pct") or usage.get("snap_share") or usage.get("avg_off_snap_pct"),
        games,
        touches,
    )

    target_share = _target_share(
        usage.get("target_share") if usage.get("target_share") is not None else usage.get("target_share_pct"),
        targets,
        games,
    )
    # target_share_pct is 0–100 in some rows.
    if target_share is not None and target_share > 1.5:
        target_share = target_share / 100.0

    air_yards = _f(usage.get("air_yards") or usage.get("receiving_air_yards"))
    adot = _f(usage.get("adot") or usage.get("avg_depth_of_target"))

    birth_date = identity.get("birth_date") or raw.get("birth_date") or raw.get("bDay") or raw.get("bday")
    age = age_as_of_season_start(birth_date, season)
    draft_year = _i(identity.get("draft_year") if identity.get("draft_year") is not None else raw.get("draft_year"))
    draft_round = _i(
        identity.get("nfl_draft_round")
        if identity.get("nfl_draft_round") is not None
        else identity.get("draft_round")
    )
    draft_pick = _i(
        identity.get("nfl_draft_pick")
        if identity.get("nfl_draft_pick") is not None
        else identity.get("draft_pick")
    )
    undrafted = bool(identity.get("undrafted"))
    first_season = _i(identity.get("first_season"))
    years_exp = years_experience_before_season(season, draft_year, first_season=first_season)

    source_schema = "legacy_totals" if usage.get("ppr_total") is not None or usage.get("gsis_id") else "sleeper_averages"

    starts = _i(usage.get("starts") or usage.get("gs"))

    row = {
        "season": int(season),
        "player_id": sleeper_id,
        "sleeper_id": sleeper_id,
        "gsis_id": gsis_id,
        "name": name,
        "position": position,
        "team": team,
        "age": age,
        "age_bucket": age_bucket(position, age),
        "years_experience": years_exp,
        "draft_year": draft_year,
        "nfl_draft_round": draft_round,
        "nfl_draft_pick": draft_pick,
        "draft_capital_bucket": draft_capital_bucket(draft_round, draft_pick, undrafted=undrafted),
        "games": _i(games) if games is not None else None,
        "starts": starts,
        "snaps": snaps,
        "snap_pct": snap_pct,
        "targets": targets,
        "target_share": target_share,
        "receptions": receptions,
        "receiving_yards": receiving_yards,
        "receiving_tds": receiving_tds,
        "air_yards": air_yards,
        "adot": adot,
        "carries": carries,
        "rush_yards": rush_yards,
        "rush_tds": rush_tds,
        "red_zone_targets": rz_targets,
        "red_zone_carries": rz_carries,
        "passing_attempts": passing_attempts,
        "passing_yards": passing_yards,
        "passing_tds": passing_tds,
        "interceptions": interceptions,
        "ppr_points": ppr_points,
        "half_ppr_points": half_points,
        "standard_points": std_points,
        "ppr_ppg": ppr_ppg,
        "half_ppr_ppg": half_ppg,
        "standard_ppg": std_ppg,
        "source_schema": source_schema,
        # Legacy avg_* columns so load_player_history_df +
        # build_player_history_features (live valuation) keep working.
        "avg_off_snap_pct": snap_pct,
        "avg_off_snaps": _per_game(snaps, games),
        "avg_targets": _per_game(targets, games),
        "avg_receptions": _per_game(receptions, games),
        "avg_rec_yards": _per_game(receiving_yards, games),
        "avg_rec_tds": _per_game(receiving_tds, games),
        "avg_carries": _per_game(carries, games),
        "avg_rush_yards": _per_game(rush_yards, games),
        "avg_rush_tds": _per_game(rush_tds, games),
        "avg_pass_att": _per_game(passing_attempts, games),
        "avg_pass_yds": _per_game(passing_yards, games),
        "avg_pass_tds": _per_game(passing_tds, games),
        "avg_pass_int": _per_game(interceptions, games),
        "rec_rz_tgt_pg": _per_game(rz_targets, games),
        "rush_rz_att_pg": _per_game(rz_carries, games),
        "std_ppg": std_ppg,
    }
    for field in EFFICIENCY_FIELDS:
        row.setdefault(field, None)
    return row


def _per_game(total: Optional[float], games: Optional[float]) -> Optional[float]:
    if total is None or games is None or games <= 0:
        return None
    return total / games


def row_appeared(row: Mapping[str, Any]) -> bool:
    """True when the player actually appeared (games or any production).

    2023–2025 Sleeper usage_rows pad hundreds of games=0 stubs that the
    2018–2022 nflverse extracts never included. Those are not seasons.
    """
    if (row.get("games") or 0) > 0:
        return True
    for key in ("ppr_points", "targets", "carries", "passing_attempts", "receptions"):
        val = row.get(key)
        if val is not None and val != 0:
            return True
    return False


def coverage_counts(rows: list) -> dict:
    """Per-field non-null counts for a list of canonical season rows."""
    n = len(rows)
    fields = {}
    for col in EFFICIENCY_COVERAGE_COLUMNS:
        present = sum(1 for r in rows if r.get(col) is not None)
        fields[col] = {"present": present, "missing": n - present, "n": n}
    return {"n": n, "fields": fields}


def identity_from_players_index_entry(meta: Mapping[str, Any]) -> dict:
    """Lift birth date / draft year / name from a ``players_index`` value."""
    if not meta:
        return {}
    pos = _str(meta.get("position") or meta.get("pos"))
    if pos:
        pos = pos.upper()
    return {
        "name": _str(meta.get("name") or meta.get("full_name")),
        "position": pos,
        "team": _str(meta.get("team")),
        "birth_date": meta.get("bDay") or meta.get("bday") or meta.get("birth_date"),
        "draft_year": meta.get("draft_year"),
        "years_exp_now": meta.get("exp") or meta.get("years_exp"),
    }
