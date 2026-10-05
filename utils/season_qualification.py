"""Shared, schedule-backed qualification rules for in-season statistics.

Qualification is deliberately based on *fully completed NFL rounds*.  A
Thursday game does not advance the sample while the rest of that week's slate
is still being played, and a bye never counts against an individual player.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable, Iterable, Optional


FULL_GAMES_MIN = 4
FULL_VOLUME_MINS = {"games": 4, "total_pass_att": 50, "total_carries": 20,
                    "total_targets": 15, "total_receptions": 10,
                    "total_touches": 20}


def _is_regular(game: dict) -> bool:
    value = str(game.get("seasonType") or game.get("season_type") or "").lower()
    return value in ("", "2", "reg", "regular", "regular season") or "regular" in value


def _is_final(game: dict) -> bool:
    """Return whether a game is safely known to be complete.

    Provider completion flags are authoritative.  The schedule cache can,
    however, retain its preseason ``Scheduled`` status after a game has been
    played.  Once the game's *calendar date* is in the past, it is also safe to
    consider it complete.  This deliberately does not use kickoff timestamps,
    so a round cannot qualify while games on its final calendar day are live.
    """
    if game.get("completed") is True or game.get("is_complete") is True:
        return True
    code = str(game.get("gameStatusCode") or game.get("status_code") or "").strip()
    status = str(game.get("gameStatus") or game.get("status") or "").lower()
    if code in ("2", "3") or "final" in status or "completed" in status:
        return True
    # A date fallback must not convert an explicitly postponed/cancelled game
    # into a final merely because its original date has passed.
    if any(word in status for word in ("postpon", "cancel", "suspend")):
        return False

    game_date = str(game.get("gameDate") or game.get("game_date") or "")[:10]
    compact = game_date.replace("-", "")
    if len(compact) != 8 or not compact.isdigit():
        return False
    try:
        scheduled_date = date.fromisoformat(
            f"{compact[:4]}-{compact[4:6]}-{compact[6:]}"
        )
    except ValueError:
        return False
    return scheduled_date < date.today()


def completed_regular_season_rounds(
    season: int,
    *,
    week_start: Optional[int] = None,
    week_end: Optional[int] = None,
    load_week: Optional[Callable[[int, int], Iterable[dict]]] = None,
    max_week: int = 18,
) -> list[int]:
    """Return rounds whose complete regular-season slate is provider-final."""
    if load_week is None:
        from utils.utils import load_week_schedule
        load_week = load_week_schedule
    lo = max(1, int(week_start or 1))
    hi = min(max_week, int(week_end or max_week))
    completed = []
    for week in range(lo, hi + 1):
        games = [g for g in (load_week(int(season), week) or [])
                 if isinstance(g, dict) and _is_regular(g)]
        if games and all(_is_final(g) for g in games):
            completed.append(week)
    return completed


def scaled_minimum(full_minimum: int, completed_rounds: int) -> int:
    """Scale cumulative gates linearly through the normal four-game sample."""
    full = max(1, int(full_minimum))
    rounds = max(0, int(completed_rounds))
    if rounds == 0:
        return 1
    return max(1, min(full, (full * min(rounds, FULL_GAMES_MIN) + FULL_GAMES_MIN - 1)
                            // FULL_GAMES_MIN))


def player_completed_weeks(
    player_id: str,
    season: int,
    *,
    load_week: Optional[Callable[[int, int], Iterable[dict]]] = None,
) -> list[int]:
    """Weeks where the player's own team game is final.

    Unlike :func:`completed_regular_season_rounds` (which needs the whole NFL
    slate final), a week counts here as soon as THAT PLAYER's team has
    finished playing.  The modal's sample note therefore updates the moment
    their game goes final, even mid-week.  Bye weeks never count against the
    player.  Mid-season trades are handled via the per-week team map.
    """
    if load_week is None:
        from utils.utils import load_week_schedule
        load_week = load_week_schedule
    season = int(season)

    # Player's team per week (handles mid-season trades).
    try:
        from data_building.external_data.player_team_history import (
            teams_in_season,
        )
        stints = teams_in_season(str(player_id), season) or []
    except Exception:
        stints = []
    if not stints:
        return []
    week_team: dict[int, str] = {}
    for stint in stints:
        team = str(stint.get("team") or "").strip().upper()
        if not team:
            continue
        weeks = stint.get("weeks") or []
        if weeks:
            for w in weeks:
                week_team.setdefault(int(w), team)
        else:
            # Season-granularity fallback: team unknown per week.
            for w in range(1, 19):
                week_team.setdefault(w, team)

    # Bound the scan: the current season stops at the current NFL week; past
    # seasons scan the full schedule (the date fallback marks old games final).
    try:
        from dashboard_services.api import get_nfl_state
        current = get_nfl_state() or {}
        cur_season = int(current.get("season") or 0)
        cur_week = int(current.get("week") or 0)
    except Exception:
        cur_season, cur_week = 0, 0
    if cur_season and season == cur_season and cur_week:
        max_week = min(18, cur_week)
    elif cur_season and season > cur_season:
        return []
    else:
        max_week = 18

    completed = []
    for week in range(1, max_week + 1):
        team = week_team.get(week)
        if not team:
            continue  # bye week or no team data; never counts against them
        try:
            games = [g for g in (load_week(season, week) or [])
                     if isinstance(g, dict)]
        except Exception:
            continue
        game = next(
            (g for g in games
             if str(g.get("away") or "").upper() == team
             or str(g.get("home") or "").upper() == team),
            None,
        )
        if game is None:
            continue
        if _is_final(game):
            completed.append(week)
            continue
        status = str(game.get("gameStatus") or game.get("status") or "").lower()
        if any(word in status for word in ("postpon", "cancel", "suspend")):
            continue  # odd scheduling; keep scanning later weeks
        # Weeks are chronological: a non-final game means later weeks have
        # not been played yet.
        break
    return completed


def player_sample_note(player_id: str, season: int) -> Optional[str]:
    """Per-player small-sample note, e.g. "Small sample · 3 games".

    Counts weeks where the player's own team game is final, so the note
    updates as soon as their game finishes.  Returns None once the sample is
    no longer small (4+ completed games) or the player has no team data.
    """
    n = len(player_completed_weeks(player_id, season))
    if 0 < n < FULL_GAMES_MIN:
        return f"Small sample · {n} game{'s' if n != 1 else ''}"
    return None


def player_qualification_note(
    player_id: str,
    season: int,
    *,
    fallback_note: Optional[str] = None,
    fallback_provisional: bool = False,
) -> tuple[Optional[str], bool]:
    """Per-player (note, provisional), falling back to league-wide values.

    The note counts weeks where the player's own team game is final, so it
    updates the moment their game finishes, even mid-week.  When per-player
    team data is unavailable (or no games played yet), returns the provided
    fallbacks (typically the league-wide policy's note/provisional).
    """
    try:
        n = len(player_completed_weeks(player_id, season))
    except Exception:
        n = 0
    if n == 0:
        return fallback_note, fallback_provisional
    if n < FULL_GAMES_MIN:
        return f"Small sample · {n} game{'s' if n != 1 else ''}", True
    return None, False


@dataclass(frozen=True)
class QualificationPolicy:
    season: int
    completed_weeks: tuple[int, ...]
    games_min: int

    @property
    def provisional(self) -> bool:
        return 0 < len(self.completed_weeks) < FULL_GAMES_MIN

    def minimum(self, volume_column: str, full_minimum: Optional[int] = None) -> int:
        normal = int(full_minimum or FULL_VOLUME_MINS.get(volume_column, 1))
        return scaled_minimum(normal, len(self.completed_weeks))

    def note(self) -> Optional[str]:
        if not self.provisional:
            return None
        n = len(self.completed_weeks)
        # Counts fully completed NFL rounds, not the player's games. Word it
        # as weeks so it can't be misread as a games-played total (the stats
        # tab may show an in-progress week the rankings don't count yet).
        return f"Small sample · {n} week{'s' if n != 1 else ''} final"


def qualification_policy(season: int, *, week_start: Optional[int] = None,
                         week_end: Optional[int] = None, load_week=None) -> QualificationPolicy:
    weeks = completed_regular_season_rounds(
        season, week_start=week_start, week_end=week_end, load_week=load_week)
    progress = len(weeks)
    # Missing old schedule files must not turn a completed historical season
    # into a one-game sample. The current season is obtained from provider state,
    # not the wall clock.
    if not weeks and load_week is None:
        try:
            from dashboard_services.api import get_nfl_state
            current = int((get_nfl_state() or {}).get("season") or season)
            if int(season) < current:
                lo = max(1, int(week_start or 1))
                hi = min(18, int(week_end or 18))
                weeks = list(range(lo, hi + 1))
                progress = FULL_GAMES_MIN
        except Exception:
            pass
    return QualificationPolicy(int(season), tuple(weeks),
                               scaled_minimum(FULL_GAMES_MIN, progress))
