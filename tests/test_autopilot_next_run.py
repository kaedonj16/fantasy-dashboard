"""Next-run computation pins for the Autopilot schedule.

Two layers:

- Slate-derived windows (the real behavior): next_run_after /
  next_run_label read the week's real NFL slate through the ``rows=``
  seam. The Thanksgiving assertions use the real 2026 Week 12 slate
  shape: a Wednesday night game, the Thursday tripleheader
  (13:00 / 16:30 / 20:20 ET), a Black Friday game, and the Sunday
  slate. On Thanksgiving morning the next run is Thu 12:00 PM ET,
  ahead of the 1:00 PM kickoff, not the old fixed Thu 7:00 PM.
- Legacy fallback: with no slate rows the fixed anchors in
  autopilot_schedule.RUN_WINDOWS_UTC answer (mirroring render.yaml's
  autopilot-thu "0 0 * * 5" UTC and autopilot-sun "30 16 * * 0" UTC):
  the Thursday pass lands Thu evening Eastern (19:00 EST / 20:00 EDT)
  and the Sunday pass Sun late morning Eastern (11:30 EST /
  12:30 EDT).
"""
from datetime import datetime, timezone

import pytest

pytest.importorskip("flask")
# autopilot_schedule's import chain pulls the plan builder (pandas).
pytest.importorskip("pandas")

from dashboard_services import autopilot_schedule as sched


def _utc(y, m, d, hh=0, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


def _row(week, gameday, gametime, away, home):
    return {
        "season": "2026", "game_type": "REG", "week": str(week),
        "gameday": gameday, "gametime": gametime,
        "away_team": away, "home_team": home,
    }


# The real 2026 Week 12 (Thanksgiving) slate, nflverse shape.
THANKSGIVING = [
    _row(12, "2026-11-25", "20:00", "GB", "LA"),
    _row(12, "2026-11-26", "13:00", "CHI", "DET"),
    _row(12, "2026-11-26", "16:30", "PHI", "DAL"),
    _row(12, "2026-11-26", "20:20", "KC", "BUF"),
    _row(12, "2026-11-27", "15:00", "DEN", "PIT"),
    _row(12, "2026-11-29", "13:00", "NO", "CIN"),
    _row(12, "2026-11-29", "13:00", "ATL", "MIN"),
    _row(12, "2026-11-29", "16:05", "TEN", "JAX"),
    _row(12, "2026-11-29", "16:25", "SEA", "SF"),
    _row(12, "2026-11-29", "20:20", "NE", "LAC"),
    _row(12, "2026-11-30", "20:15", "CAR", "TB"),
]


# ── Slate-derived ──────────────────────────────────────────────────────

def test_next_run_thanksgiving_morning_is_noon_not_7pm():
    # Thu 08:00 EST: the next run serves the 13:00 kickoff.
    nxt = sched.next_run_after(_utc(2026, 11, 26, 13, 0), rows=THANKSGIVING)
    assert (nxt.year, nxt.month, nxt.day) == (2026, 11, 26)
    assert (nxt.hour, nxt.minute) == (12, 0)
    assert sched.next_run_label(_utc(2026, 11, 26, 13, 0), rows=THANKSGIVING) == \
        "Today, Thu 12:00 PM ET"


def test_next_run_wednesday_game_gets_a_wednesday_evening_run():
    # Wed 12:00 EST: a game kicks off tonight (20:00), so the next run
    # is tonight at 19:00, not Thursday.
    nxt = sched.next_run_after(_utc(2026, 11, 25, 17, 0), rows=THANKSGIVING)
    assert (nxt.month, nxt.day, nxt.hour, nxt.minute) == (11, 25, 19, 0)
    assert sched.next_run_label(_utc(2026, 11, 25, 17, 0), rows=THANKSGIVING) == \
        "Today, Wed 7:00 PM ET"


def test_next_run_inside_a_due_window_is_that_window():
    # Thu 12:05 EST: the noon window is due right now (kickoff 13:00).
    nxt = sched.next_run_after(_utc(2026, 11, 26, 17, 5), rows=THANKSGIVING)
    assert (nxt.month, nxt.day, nxt.hour, nxt.minute) == (11, 26, 12, 0)


def test_next_run_after_friday_kickoff_rolls_to_sunday():
    # Fri 15:30 EST: the Black Friday game has kicked off; the week's
    # remaining run is the Sunday late-morning pass.
    nxt = sched.next_run_after(_utc(2026, 11, 27, 20, 30), rows=THANKSGIVING)
    assert (nxt.month, nxt.day, nxt.hour, nxt.minute) == (11, 29, 11, 30)
    assert sched.next_run_label(_utc(2026, 11, 27, 20, 30), rows=THANKSGIVING) == \
        "Sun, Nov 29 · 11:30 AM ET"


# ── Legacy fallback (no slate rows) ────────────────────────────────────

def test_fallback_thursday_window_from_wednesday():
    nxt = sched.next_run_after(_utc(2026, 11, 25, 17, 0), rows=[])
    assert (nxt.year, nxt.month, nxt.day) == (2026, 11, 26)
    assert (nxt.hour, nxt.minute) == (19, 0)  # Thu 19:00 EST
    assert sched.next_run_label(_utc(2026, 11, 25, 17, 0), rows=[]) == \
        "Tomorrow, Thu 7:00 PM ET"


def test_fallback_sunday_window_after_thursday_passed():
    nxt = sched.next_run_after(_utc(2026, 11, 27, 0, 30), rows=[])
    assert (nxt.year, nxt.month, nxt.day) == (2026, 11, 29)
    assert (nxt.hour, nxt.minute) == (11, 30)  # Sun 11:30 EST
    assert sched.next_run_label(_utc(2026, 11, 27, 0, 30), rows=[]) == \
        "Sun, Nov 29 · 11:30 AM ET"


def test_fallback_exactly_at_window_rolls_to_next_window():
    nxt = sched.next_run_after(_utc(2026, 11, 27, 0, 0), rows=[])
    assert (nxt.month, nxt.day) == (11, 29)  # Sunday, not next Thursday


def test_fallback_sunday_same_day():
    nxt = sched.next_run_after(_utc(2026, 11, 29, 16, 0), rows=[])
    assert (nxt.month, nxt.day, nxt.hour, nxt.minute) == (11, 29, 11, 30)
    assert sched.next_run_label(_utc(2026, 11, 29, 16, 0), rows=[]) == \
        "Today, Sun 11:30 AM ET"


def test_fallback_week_boundary_sunday_evening_to_thursday():
    nxt = sched.next_run_after(_utc(2026, 11, 29, 17, 0), rows=[])
    assert (nxt.year, nxt.month, nxt.day) == (2026, 12, 3)
    assert (nxt.hour, nxt.minute) == (19, 0)  # Thu 19:00 EST
    assert sched.next_run_label(_utc(2026, 11, 29, 17, 0), rows=[]) == \
        "Thu, Dec 3 · 7:00 PM ET"


def test_fallback_dst_summer_times():
    # EDT: the Thursday pass is 20:00 local, the Sunday pass 12:30.
    nxt = sched.next_run_after(_utc(2026, 7, 1, 12, 0), rows=[])
    assert (nxt.month, nxt.day, nxt.hour, nxt.minute) == (7, 2, 20, 0)
    assert sched.next_run_label(_utc(2026, 7, 1, 12, 0), rows=[]) == \
        "Tomorrow, Thu 8:00 PM ET"
    nxt = sched.next_run_after(_utc(2026, 7, 4, 12, 0), rows=[])
    assert (nxt.month, nxt.day, nxt.hour, nxt.minute) == (7, 5, 12, 30)
    assert sched.next_run_label(_utc(2026, 7, 4, 12, 0), rows=[]) == \
        "Tomorrow, Sun 12:30 PM ET"
