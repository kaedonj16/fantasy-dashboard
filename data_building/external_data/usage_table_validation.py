"""Usage-table completeness checks.

Kept as a stdlib-only module so the unit suite can cover the offseason /
pre-kickoff / in-season branches without importing pandas or Sleeper clients.
"""
from __future__ import annotations

from typing import Dict, List, Optional


def validate_usage_table(
    players_out: List[dict],
    usage_by_pid: Dict[str, dict],
    season: int,
    nfl_state: Optional[dict] = None,
) -> None:
    """Raise ValueError if the usage table looks like a failed/incomplete fetch.

    0 games is expected until regular-season games have actually been played:
    true offseason ("off"), preseason ("pre"), AND the stretch after Sleeper
    flips season_type to "regular" / week to 1 but before Thursday kickoff
    (often a week+ early). Treating that last case as in-season used to reject
    the (correct) all-zero-games table, which stopped usage_table.json from
    being written and froze player values on the last good model_values.json.

    ``usage_by_pid`` and ``season`` are kept on the signature for call-site
    compatibility; they are not currently used.
    """
    del usage_by_pid, season  # reserved for future checks

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
    with_usage = sum(1 for p in players_out if p.get("usage", {}).get("ppr_ppg", 0) > 0)

    no_games_expected = season_type in ("off", "pre")
    # Week 1 with a table that would fail the in-season checks is the
    # pre-kickoff (or stats-not-in-yet) snapshot, not a broken fetch.
    # Mid-season (week 2+) still uses the strict checks below.
    if (
        not no_games_expected
        and week <= 1
        and (zero_games_pct > 0.6 or with_usage < 200)
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

    if zero_games_pct > 0.6:
        raise ValueError(
            f"[VALIDATION ERROR] Too many players with 0 games: {zero_games}/{total_players} "
            f"({zero_games_pct:.1%}). Data fetch likely incomplete. (Season type: {season_type})"
        )

    if with_usage < 200:
        raise ValueError(
            f"[VALIDATION ERROR] Too few players with production: {with_usage} "
            f"(expected 300+). Usage data may be missing. (Season type: {season_type})"
        )

    print("[VALIDATION OK] In-season usage table validated:")
    print(f"  - Total players: {total_players}")
    print(f"  - Players with 0 games: {zero_games} ({zero_games_pct:.1%})")
    print(f"  - Players with production: {with_usage}")
