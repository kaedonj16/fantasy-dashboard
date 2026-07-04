"""Tests for utils.relative_time (human-relative timestamps)."""
from datetime import datetime, timedelta

from utils.relative_time import EASTERN, rel_time

# Fixed reference: a Wednesday afternoon in Eastern time.
NOW = datetime(2026, 7, 1, 15, 0, 0, tzinfo=EASTERN)


def test_just_now():
    assert rel_time(NOW - timedelta(seconds=30), now=NOW) == "Just now"


def test_minutes_ago():
    assert rel_time(NOW - timedelta(minutes=5), now=NOW) == "5m ago"
    assert rel_time(NOW - timedelta(minutes=59), now=NOW) == "59m ago"


def test_earlier_today():
    assert rel_time(NOW - timedelta(hours=3), now=NOW) == "Today 12:00 PM"


def test_today_strips_leading_zero_hour():
    dt = datetime(2026, 7, 1, 9, 5, 0, tzinfo=EASTERN)
    assert rel_time(dt, now=NOW) == "Today 9:05 AM"


def test_yesterday():
    dt = datetime(2026, 6, 30, 20, 0, 0, tzinfo=EASTERN)
    assert rel_time(dt, now=NOW) == "Yesterday"


def test_days_ago():
    dt = datetime(2026, 6, 28, 12, 0, 0, tzinfo=EASTERN)
    assert rel_time(dt, now=NOW) == "3d ago"


def test_weeks_ago():
    dt = NOW - timedelta(days=14)
    assert rel_time(dt, now=NOW) == "2w ago"


def test_beyond_a_month_shows_date():
    dt = datetime(2026, 5, 12, 12, 0, 0, tzinfo=EASTERN)
    assert rel_time(dt, now=NOW) == "May 12"


def test_converts_other_timezones():
    from zoneinfo import ZoneInfo
    utc_dt = (NOW - timedelta(minutes=10)).astimezone(ZoneInfo("UTC"))
    assert rel_time(utc_dt, now=NOW) == "10m ago"
