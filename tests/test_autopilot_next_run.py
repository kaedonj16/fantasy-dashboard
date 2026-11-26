"""Next-run computation pins for the Autopilot schedule.

The windows are the app's single source of truth in
autopilot_schedule.RUN_WINDOWS_UTC, mirroring render.yaml's cron
services (autopilot-thu "0 0 * * 5" UTC, autopilot-sun "30 16 * * 0"
UTC): the Thursday pass lands Thu evening Eastern (19:00 EST /
20:00 EDT) and the Sunday pass Sun late morning Eastern (11:30 EST /
12:30 EDT). These tests pin both windows, the week boundary, and the
DST-dependent Eastern display times.
"""
from datetime import datetime, timezone

import pytest

pytest.importorskip("flask")
# autopilot_schedule's import chain pulls the plan builder (pandas).
pytest.importorskip("pandas")

from dashboard_services import autopilot_schedule as sched


def _utc(y, m, d, hh=0, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


def test_next_run_thursday_window_from_wednesday():
    nxt = sched.next_run_after(_utc(2026, 11, 25, 17, 0))  # Wed 12:00 EST
    assert (nxt.year, nxt.month, nxt.day) == (2026, 11, 26)
    assert (nxt.hour, nxt.minute) == (19, 0)  # Thu 19:00 EST
    assert sched.next_run_label(_utc(2026, 11, 25, 17, 0)) == \
        "Tomorrow, Thu 7:00 PM ET"


def test_next_run_sunday_window_after_thursday_passed():
    nxt = sched.next_run_after(_utc(2026, 11, 27, 0, 30))  # just after Thu run
    assert (nxt.year, nxt.month, nxt.day) == (2026, 11, 29)
    assert (nxt.hour, nxt.minute) == (11, 30)  # Sun 11:30 EST
    assert sched.next_run_label(_utc(2026, 11, 27, 0, 30)) == \
        "Sun, Nov 29 · 11:30 AM ET"


def test_next_run_exactly_at_window_rolls_to_next_window():
    nxt = sched.next_run_after(_utc(2026, 11, 27, 0, 0))  # exactly Thu run
    assert (nxt.month, nxt.day) == (11, 29)  # Sunday, not next Thursday


def test_next_run_sunday_same_day():
    nxt = sched.next_run_after(_utc(2026, 11, 29, 16, 0))  # 30 min before
    assert (nxt.month, nxt.day, nxt.hour, nxt.minute) == (11, 29, 11, 30)
    assert sched.next_run_label(_utc(2026, 11, 29, 16, 0)) == \
        "Today, Sun 11:30 AM ET"


def test_next_run_week_boundary_sunday_evening_to_thursday():
    nxt = sched.next_run_after(_utc(2026, 11, 29, 17, 0))  # after Sun run
    assert (nxt.year, nxt.month, nxt.day) == (2026, 12, 3)
    assert (nxt.hour, nxt.minute) == (19, 0)  # Thu 19:00 EST
    assert sched.next_run_label(_utc(2026, 11, 29, 17, 0)) == \
        "Thu, Dec 3 · 7:00 PM ET"


def test_next_run_dst_summer_times():
    # EDT: the Thursday pass is 20:00 local, the Sunday pass 12:30.
    nxt = sched.next_run_after(_utc(2026, 7, 1, 12, 0))  # Wed 08:00 EDT
    assert (nxt.month, nxt.day, nxt.hour, nxt.minute) == (7, 2, 20, 0)
    assert sched.next_run_label(_utc(2026, 7, 1, 12, 0)) == \
        "Tomorrow, Thu 8:00 PM ET"
    nxt = sched.next_run_after(_utc(2026, 7, 4, 12, 0))  # Sat 08:00 EDT
    assert (nxt.month, nxt.day, nxt.hour, nxt.minute) == (7, 5, 12, 30)
    assert sched.next_run_label(_utc(2026, 7, 4, 12, 0)) == \
        "Tomorrow, Sun 12:30 PM ET"
