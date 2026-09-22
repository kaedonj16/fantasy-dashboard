"""Usage-table completeness checks.

Kept as a stdlib-only module so the unit suite can cover the offseason /
pre-kickoff / in-season branches without importing pandas or Sleeper clients.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence


def validate_usage_table(
    players_out: List[dict],
    usage_by_pid: Dict[str, dict],
    season: int,
    nfl_state: Optional[dict] = None,
    week_diagnostics: Optional[Sequence[dict]] = None,
) -> None:
    """Raise ValueError if the usage table looks like a failed/incomplete fetch.

    0 games is expected until regular-season games have actually been played:
    true offseason ("off"), preseason ("pre"), AND the stretch after Sleeper
    flips season_type to "regular" / week to 1 but before Thursday kickoff
    (often a week+ early). Treating that last case as in-season used to reject
    the (correct) all-zero-games table, which stopped usage_table.json from
    being written and froze player values on the last good model_values.json.

    Historical/inactive rows deliberately remain in ``players_out`` for model
    compatibility, but are never the in-season completeness denominator.
    Completeness is measured from the requested source weeks and the distinct
    current-season players present in ``usage_by_pid``.
    """
    del season

    total_players = len(players_out)

    # Basic size check (always applies)
    if total_players < 400:
        raise ValueError(
            f"[VALIDATION ERROR] Usage table too small: {total_players} players "
            f"(expected 500+). Sleeper API may have failed."
        )

    nfl_state = nfl_state or {}
    season_type = str(nfl_state.get("season_type", "")).lower().strip()
    try:
        week = int(nfl_state.get("week") or 0)
    except (TypeError, ValueError):
        week = 0

    zero_games = sum(1 for p in players_out if p.get("usage", {}).get("games", 0) == 0)
    zero_games_pct = zero_games / total_players if total_players > 0 else 0
    cohort = [p for p in players_out if (not usage_by_pid or str(p.get("id")) in usage_by_pid)
              and (p.get("usage") or {}).get("games", 0) > 0]
    with_games = len(cohort)
    with_usage = sum(1 for p in cohort if any(
        float((p.get("usage") or {}).get(k) or 0) != 0
        for k in ("ppr_ppg", "targets", "carries", "avg_pass_att", "avg_targets", "avg_carries")
    ))
    positions = {str(p.get("position") or "").upper() for p in cohort if p.get("position")}

    no_games_expected = season_type in ("off", "pre")
    # Week 1/2 with a table that would fail the in-season checks is the
    # pre-kickoff (or stats-not-in-yet) snapshot, not a broken fetch.
    # Sleeper often bumps "week" to 2 while Week 1 games are still in
    # progress (e.g. Monday night), so stats may not have propagated yet.
    # By week 3 every team has played at least once; strict checks apply.
    if (
        not no_games_expected
        and week <= 2
        and (with_games < 100 or with_usage < 50)
    ):
        no_games_expected = True

    if no_games_expected:
        print("[VALIDATION OK] No regular-season games yet - usage table validated:")
        print(f"  - Season type: {season_type or '(none)'}, week: {week}")
        print(f"  - Total players: {total_players}")
        print(
            f"  - Players with 0 games: {zero_games} ({zero_games_pct:.1%}) "
            "[EXPECTED before games]"
        )
        print(
            f"  - Players with production: {with_usage} "
            "[Most should be 0 before games]"
        )
        return

    completed = [d for d in (week_diagnostics or []) if d.get("phase") == "completed"]
    if completed:
        missing = [d.get("week") for d in completed
                   if d.get("response_type") != "dict" or int(d.get("raw_row_count") or 0) == 0]
        if missing:
            raise ValueError(f"[VALIDATION ERROR] Missing/corrupt completed weeks: {missing}")

    # Week-aware conservative floors: by week 3 a healthy feed has hundreds of
    # distinct participants, while an early completed week can be much smaller.
    min_players = 100 if week <= 3 else 180
    min_production = 50 if week <= 3 else 100
    if with_games < min_players:
        raise ValueError(
            f"[VALIDATION ERROR] Too many players with 0 games / too few current-season players with games: {with_games} "
            f"(minimum {min_players}; season type: {season_type}, week: {week})"
        )

    if with_usage < min_production:
        raise ValueError(
            f"[VALIDATION ERROR] Too few players with production: {with_usage} "
            f"(minimum {min_production}). Usage data may be missing. (Season type: {season_type})"
        )
    if positions and not {"QB", "RB", "WR", "TE"}.issubset(positions):
        raise ValueError(f"[VALIDATION ERROR] Missing core position coverage: {sorted(positions)}")

    print("[VALIDATION OK] In-season usage table validated:")
    print(f"  - Total players: {total_players}")
    print(f"  - Players with 0 games: {zero_games} ({zero_games_pct:.1%})")
    print(f"  - Players with production: {with_usage}")
    print(f"  - Current-season players with games: {with_games}")
    print(f"  - Position coverage: {sorted(positions)}")
